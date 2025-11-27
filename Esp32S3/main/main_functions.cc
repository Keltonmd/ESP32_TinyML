#include "main_functions.h"      // Cabeçalho principal do projeto
#include "mqtt_client.h"         // Cliente MQTT do ESP-IDF
#include "tensorflow/lite/micro/micro_interpreter.h"  // TensorFlow Lite Micro Interpreter
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h" // Resolver de operações TFLM
#include "tensorflow/lite/schema/schema_generated.h" // Esquema do modelo TFLite

#include "esp_wifi.h"            // WiFi ESP-IDF
#include "esp_event.h"           // Eventos do ESP-IDF
#include "esp_netif.h"           // Rede TCP/IP do ESP-IDF
#include "esp_log.h"             // Logging do ESP-IDF
#include "nvs_flash.h"           // Non-Volatile Storage
#include "model.h"               // Modelo TFLite convertido para C array

#include <cstring>
#include <cmath>
#include <cstdio>

// ==========================================================================
// 1. Variáveis globais do TFLM e MQTT
// ==========================================================================
namespace {
    const tflite::Model* model = nullptr;          // Ponteiro para o modelo carregado
    tflite::MicroInterpreter* tflInterpreter = nullptr; // Interpreter TFLM
    TfLiteTensor* tflInputTensor  = nullptr;      // Tensor de entrada
    TfLiteTensor* tflOutputTensor = nullptr;      // Tensor de saída

    float   tfInputScale       = 0.0f;            // Escala para quantização de entrada
    int32_t tfInputZeroPoint   = 0;               // Zero-point de entrada
    float   tfOutputScale      = 0.0f;            // Escala para quantização de saída
    int32_t tfOutputZeroPoint  = 0;               // Zero-point de saída

    constexpr int kImageSize = 32 * 32 * 3;       // Tamanho da imagem de entrada (32x32 RGB)
    constexpr int kTensorArenaSize = 64 * 1024;   // Tamanho do arena de tensores (64 KB)
    uint8_t tensor_arena[kTensorArenaSize];       // Arena de memória para TFLM

    esp_mqtt_client_handle_t mqtt_client = nullptr; // Handle do cliente MQTT

    bool finalizar = false;                        // Flag para encerrar o loop
}

// ==========================================================================
// Função de callback para eventos do WiFi
// ==========================================================================
static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                               int32_t event_id, void* event_data)
{
    // Evento: WiFi começou
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();                        // Conecta ao AP
        printf("Tentando conectar...\n");
    }
    // Evento: Conectado ao AP
    else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_CONNECTED) {
        printf("Conectado ao WiFi!\n");
    }
    // Evento: Desconectado do AP
    else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        printf("WiFi desconectado! Tentando reconectar...\n");
        esp_wifi_connect();
    }
    // Evento: Obteve IP
    else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;

        printf("   IP obtido:\n");
        printf("   IP      : " IPSTR "\n", IP2STR(&event->ip_info.ip));
        printf("   Máscara : " IPSTR "\n", IP2STR(&event->ip_info.netmask));
        printf("   Gateway : " IPSTR "\n", IP2STR(&event->ip_info.gw));

        printf("   Rede configurada com sucesso!\n");
    }
}

// ==========================================================================
// Inicializa o WiFi em modo Station
// ==========================================================================
void wifi_init_sta() {
    esp_err_t ret = nvs_flash_init();            // Inicializa memória flash
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());      // Apaga NVS se necessário
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    esp_netif_init();                             // Inicializa TCP/IP stack
    esp_event_loop_create_default();              // Cria loop de eventos

    esp_netif_create_default_wifi_sta();          // Cria interface WiFi STA

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT(); // Config default
    esp_wifi_init(&cfg);

    // Registra handlers de eventos
    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL);

    // Configurações da rede WiFi
    wifi_config_t wifi_config = {};
    strcpy((char*)wifi_config.sta.ssid,     "NomeDaRedeWiFi");
    strcpy((char*)wifi_config.sta.password, "senhaWiFi123");
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    esp_wifi_set_mode(WIFI_MODE_STA);             // Define modo Station
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();                             // Inicia WiFi

    printf("Conectando ao WiFi SSID: %s...\n", wifi_config.sta.ssid);
}

// ==========================================================================
// Inicializa TensorFlow Lite Micro
// ==========================================================================
bool inicializar_tflm() {

    model = tflite::GetModel(model_tflite);       // Carrega modelo TFLite
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Modelo incompatível!\n");
        return false;
    }

    static tflite::MicroMutableOpResolver<8> resolver; // Resolver de 8 operações
    resolver.AddConv2D();
    resolver.AddLeakyRelu();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddLogistic();
    resolver.AddQuantize();
    resolver.AddDequantize();

    // Cria o interpreter estático
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    tflInterpreter = &static_interpreter;

    if (tflInterpreter->AllocateTensors() != kTfLiteOk) { // Aloca tensores
        printf("Falha ao alocar tensores!\n");
        return false;
    }

    tflInputTensor  = tflInterpreter->input(0);  // Tensor de entrada
    tflOutputTensor = tflInterpreter->output(0); // Tensor de saída

    // Obtém parâmetros de quantização
    tfInputScale      = tflInputTensor->params.scale;
    tfInputZeroPoint  = tflInputTensor->params.zero_point;
    tfOutputScale     = tflOutputTensor->params.scale;
    tfOutputZeroPoint = tflOutputTensor->params.zero_point;

    printf("TFLM inicializado.\n");
    return true;
}

// ==========================================================================
// Executa predição do modelo TFLite
// ==========================================================================
int predicao(const int8_t* input_data)
{
    memcpy(tflInputTensor->data.int8, input_data, kImageSize); // Copia imagem para tensor

    if (tflInterpreter->Invoke() != kTfLiteOk) { // Executa inferência
        printf("Falha na inferência!\n");
        return -1;
    }

    int8_t q = tflOutputTensor->data.int8[0];          // Resultado quantizado
    float out = (q - tfOutputZeroPoint) * tfOutputScale; // Dequantiza
    out = roundf(out);

    printf("Resultado = %.1f\n", out);
    return (int)out;
}

// ==========================================================================
// Callback de mensagens MQTT
// ==========================================================================
void on_message(const char* topic, int topic_len,
                const uint8_t* data, int data_len)
{
    const char* TOPICO_CLASSIFICAR = "/esp/classificar";
    const char* TOPICO_FIM         = "/colaboracao/fim";

    // Se for tópico de classificação
    if (topic_len == strlen(TOPICO_CLASSIFICAR) &&
        strncmp(topic, TOPICO_CLASSIFICAR, topic_len) == 0)
    {
        if (data_len != kImageSize) { // Verifica tamanho da imagem
            printf("Tamanho inválido (%d). Esperado %d bytes\n", data_len, kImageSize);
            return;
        }

        printf("Imagem recebida, rodando inferência...\n");
        int resultado = predicao((const int8_t*) data);

        if (resultado != -1) {
            // Publica resultado no MQTT
            esp_mqtt_client_publish(
                mqtt_client,
                "/esp/resultado",
                (char *)&resultado,
                sizeof(int),
                0, 1);

            printf("Resultado enviado: %d\n", resultado);
        }
    }
    // Se for pedido de finalizar
    else if (topic_len == strlen(TOPICO_FIM) &&
             strncmp(topic, TOPICO_FIM, topic_len) == 0)
    {
        printf("Finalizar pedido.\n");
        finalizar = true;
    }
}

// ==========================================================================
// Handler de eventos MQTT do ESP-IDF
// ==========================================================================
static void mqtt_event_handler(void* handler_args, esp_event_base_t base,
                               int32_t event_id, void* event_data)
{
    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t) event_data;

    switch ((esp_mqtt_event_id_t) event_id)
    {
        case MQTT_EVENT_CONNECTED: // Conectou ao broker
            printf("MQTT conectado.\n");
            esp_mqtt_client_subscribe(event->client, "/esp/classificar", 0);
            esp_mqtt_client_subscribe(event->client, "/colaboracao/fim", 0);
            break;

        case MQTT_EVENT_DATA: // Recebeu mensagem
            on_message(
                event->topic,
                event->topic_len,
                (const uint8_t*) event->data,
                event->data_len);
            break;

        default:
            break;
    }
}

// ==========================================================================
// Setup principal
// ==========================================================================
void setup() {
    wifi_init_sta();        // Inicializa WiFi
    inicializar_tflm();     // Inicializa TFLite Micro

    // Configura cliente MQTT
    esp_mqtt_client_config_t cfg = {};
    cfg.broker.address.uri = "mqtt://enderecoIP:1883";
    cfg.session.keepalive = 60;
    cfg.buffer.size = 4096;
    cfg.network.disable_auto_reconnect = false;
    cfg.network.timeout_ms = 5000;

    mqtt_client = esp_mqtt_client_init(&cfg);                          // Inicializa MQTT
    esp_mqtt_client_register_event(mqtt_client, MQTT_EVENT_ANY, mqtt_event_handler, NULL); // Registra handler
    esp_mqtt_client_start(mqtt_client);                                // Inicia cliente

    printf("Setup concluído!\n");
}

// ==========================================================================
// Loop principal
// ==========================================================================
void loop() {
    if (finalizar) {                          // Se recebeu comando de finalizar
        printf("Loop encerrado.\n");
        esp_mqtt_client_stop(mqtt_client);    // Para MQTT
        esp_mqtt_client_destroy(mqtt_client); // Destrói cliente
        while (true) vTaskDelay(1000 / portTICK_PERIOD_MS); // Loop infinito
    }

    vTaskDelay(10 / portTICK_PERIOD_MS);      // Pequena espera para não travar o loop
}

import json

# --- CONFIGURAÇÃO ---
NOME_ARQUIVO = "CNN_Lite.ipynb" # Coloque o nome do seu arquivo aqui
MAX_LINHAS = 50  # Quantas linhas você quer manter (ex: 10 no começo + 10 no fim)
# --------------------

def encurtar_outputs(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    alterado = False

    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            if 'outputs' in cell:
                for output in cell['outputs']:
                    # Verifica se é um output de texto (stream/stdout)
                    if output.get('output_type') == 'stream' and output.get('name') == 'stdout':
                        texto = output['text']
                        
                        # Se o texto for uma lista de strings e for muito grande
                        if isinstance(texto, list) and len(texto) > MAX_LINHAS:
                            metade = MAX_LINHAS // 2
                            # Cria o novo output: Começo + Aviso + Fim
                            novo_texto = (
                                texto[:metade] + 
                                [f"\n... [ {len(texto) - MAX_LINHAS} LINHAS OCULTAS PARA VISUALIZAÇÃO NO GITHUB ] ...\n"] + 
                                texto[-metade:]
                            )
                            output['text'] = novo_texto
                            alterado = True

    if alterado:
        # Salva em um novo arquivo para segurança
        novo_nome = caminho_arquivo.replace('.ipynb', '_limpo.ipynb')
        with open(novo_nome, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"Sucesso! Novo arquivo criado: {novo_nome}")
        print("Verifique se está tudo ok e depois renomeie/substitua o original.")
    else:
        print("Nenhum output longo foi encontrado para encurtar.")

if __name__ == "__main__":
    encurtar_outputs(NOME_ARQUIVO)
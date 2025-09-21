# Processador de Imagens RGB - Trabalho Prático ICC453

Este projeto implementa operações básicas de processamento de imagens RGB usando Python, OpenCV e NumPy. O sistema processa imagens automaticamente e gera resultados organizados em diferentes categorias.

## 📋 Funcionalidades

### 1. **Ajuste de Brilho**
- Aumenta e diminui o brilho da imagem
- Gera imagens com brilho +40 e -40

### 2. **Negativo**
- Cria o negativo da imagem original

### 3. **Histograma Global**
- Calcula histograma concatenado dos canais RGB
- Salva vetor numérico em arquivo `.txt`
- Gera visualização gráfica com 4 subplots:
  - Histograma do canal Red
  - Histograma do canal Green
  - Histograma do canal Blue
  - Histograma combinado RGB

### 4. **Histograma Local**
- Divide a imagem em 6 partições (2x3)
- Calcula histograma para cada partição
- Salva vetor concatenado em arquivo `.txt`
- Gera visualização gráfica com histogramas de cada partição

### 5. **Transformações Radiométricas**
- **Linear**: Ajuste de contraste linear
- **Gamma**: Compressão/expansão com γ=0.5
- **Dente de Serra**: Efeito periódico
- **Logarítmica**: Transformação logarítmica

### 6. **Filtros e Ruído**
- **Ruído Sal e Pimenta**: Adiciona ruído à imagem
- **Filtro da Média**: Suavização por média
- **Filtro K-Vizinhos**: Filtro baseado em distância
- **Filtro da Mediana**: Remove ruído preservando bordas
- **Filtro da Moda**: Usa valor mais frequente

### 7. **Detecção de Bordas**
- **Sobel**: Detecção de bordas com gradiente
- **Canny**: Detecção de bordas com supressão de não-máximos
- Gera imagens com fundo branco e sobrepostas na original

### 8. **Descritor BIC (Border/Interior Classification)**
- Quantiza a imagem para reduzir cores
- Classifica pixels como borda ou interior
- Calcula histogramas separados para bordas e interior
- Gera imagens separadas mostrando apenas bordas ou interior

## 🚀 Instalação e Uso

### Pré-requisitos
- Python 3.7+
- pip (gerenciador de pacotes Python)

### Instalação
1. Clone o repositório:
```bash
git clone <url-do-repositorio>
cd trabalho_pratico_1-ICC453
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

### Execução
1. Coloque suas imagens na pasta `imagens_raw/`
2. Execute o programa:
```bash
python main.py
```

## 📁 Estrutura do Projeto

```
trabalho_pratico_1-ICC453/
├── imagens_raw/              # Imagens de entrada
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── img3.jpg
│   └── img4.jpg
├── resultados/               # Resultados organizados
│   ├── 01_brilho/
│   ├── 02_negativo/
│   ├── 03_histograma_global/
│   ├── 04_histograma_local/
│   ├── 05_transformacoes_radiometricas/
│   ├── 06_filtros_e_ruido/
│   ├── 07_deteccao_bordas/
│   └── 08_descritor_bic/
├── main.py                   # Código principal
├── requirements.txt          # Dependências
├── .gitignore               # Arquivos ignorados pelo Git
└── README.md                # Este arquivo
```

## 📊 Formatos de Saída

### Arquivos de Imagem
- **PNG**: Todas as imagens processadas
- **Resolução**: Mantém resolução original
- **Qualidade**: Alta qualidade sem compressão

### Arquivos de Dados
- **TXT**: Histogramas e descritores numéricos
- **Formato**: Valores inteiros separados por quebra de linha
- **Codificação**: UTF-8

### Visualizações
- **Histogramas**: Gráficos PNG com 150 DPI
- **Layout**: Organizados em subplots
- **Cores**: Canais RGB em cores correspondentes

## 🔧 Dependências

- **opencv-python**: Processamento de imagens
- **numpy**: Operações matemáticas e arrays
- **matplotlib**: Visualização de histogramas

## 📝 Exemplo de Uso

```python
# O programa processa automaticamente todas as imagens
# encontradas na pasta imagens_raw/

# Exemplo de saída:
# === Processador de Imagens RGB ===
# Buscando imagens na pasta imagens_raw...
# Encontradas 4 imagem(ns):
#    - imagens_raw\img1.jpg
#    - imagens_raw\img2.jpg
#    - imagens_raw\img3.jpg
#    - imagens_raw\img4.jpg
# Processando: img1
# img1 processada com sucesso!
# ...
# Processamento concluído!
```

## 🎯 Características Técnicas

### Algoritmos Implementados
- **Quantização**: Redução de cores para análise BIC
- **Filtros Espaciais**: Média, mediana, moda, k-vizinhos
- **Detecção de Bordas**: Sobel e Canny
- **Transformações**: Linear, gamma, logarítmica, dente de serra

### Otimizações
- **Processamento em lote**: Todas as imagens processadas automaticamente
- **Organização**: Resultados separados por categoria
- **Eficiência**: Uso otimizado de NumPy para operações vetorizadas

## 🐛 Solução de Problemas

### Erro: "Pasta 'imagens_raw' não encontrada"
- Certifique-se de que a pasta `imagens_raw/` existe
- Coloque pelo menos uma imagem na pasta

### Erro: "Nenhuma imagem encontrada"
- Verifique se as imagens estão nos formatos suportados:
  - JPG, JPEG, PNG, BMP, TIFF, TIF
- Verifique se os arquivos não estão corrompidos

### Erro de memória
- Para imagens muito grandes, considere redimensionar
- O programa usa quantização para otimizar o processamento BIC

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos como parte do curso ICC453.

## 👨‍💻 Autor

Desenvolvido como trabalho prático da disciplina ICC453 - Processamento de Imagens.

---

**Nota**: Este projeto foi desenvolvido para fins educacionais e demonstra várias técnicas fundamentais de processamento de imagens digitais.

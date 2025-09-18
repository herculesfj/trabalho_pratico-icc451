"""
Prática: Operações básicas em imagens (RGB)
Feito em Python usando OpenCV + NumPy

Uso: python main.py

O programa faz:
1) Muda o brilho da imagem
2) Faz o negativo
3) Calcula histograma global
4) Calcula histograma por partes (local)
5) Faz transformações radiométricas (linear, compressão/expansão, dente de serra, log)
6) Aplica filtros (média, vizinhos, mediana, moda) e adiciona ruído sal e pimenta
7) Detecta bordas com Sobel e Canny
8) Calcula descritor BIC (Border/Interior Classification)
"""

import os
import cv2
import numpy as np
import glob

# ---------------------- Funções auxiliares ----------------------

def ensure_dir(d):
    # Cria pasta se não existir
    if not os.path.exists(d):
        os.makedirs(d)

def save_image(path, img):
    # Salva imagem no caminho
    cv2.imwrite(path, img)

def clamp(img):
    # Garante que valores fiquem entre 0 e 255
    return np.clip(img, 0, 255).astype(np.uint8)

def find_images_in_raw():
    # Busca todas as imagens na pasta imagens_raw
    folder = "imagens_raw"
    if not os.path.exists(folder):
        print(f"Pasta '{folder}' não encontrada!")
        return []

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(folder, ext)))
        images.extend(glob.glob(os.path.join(folder, ext.upper())))
    return images

# ---------------------- 1. Brilho ----------------------

def change_brightness(img, delta):
    # Muda o brilho somando valor (delta)
    out = img.astype(np.float32) + delta
    return clamp(out)

# ---------------------- 2. Negativo ----------------------

def negative(img):
    # Faz o negativo da imagem
    return 255 - img

# ---------------------- 3. Histograma global ----------------------

def histogram_global(img):
    # Junta histogramas R, G, B (256 bins cada)
    h = []
    for ch in range(3):
        hist, _ = np.histogram(img[:,:,ch].ravel(), bins=256, range=(0,255))
        h.append(hist)
    return np.concatenate(h)

# ---------------------- 4. Histograma local ----------------------

def partition_image(img, rows, cols):
    # Divide imagem em partes (linhas x colunas)
    h, w = img.shape[:2]
    rs = np.linspace(0, h, rows+1, dtype=int)
    cs = np.linspace(0, w, cols+1, dtype=int)
    parts = []
    for i in range(rows):
        for j in range(cols):
            parts.append(img[rs[i]:rs[i+1], cs[j]:cs[j+1]])
    return parts

def histogram_local(img, partitions=(1,3)):
    # Faz histograma em cada parte da imagem
    rows, cols = partitions
    parts = partition_image(img, rows, cols)
    vecs = []
    for p in parts:
        vecs.append(histogram_global(p))
    return np.concatenate(vecs)

# ---------------------- 5. Transformadas radiométricas ----------------------

def radiometric_linear_expand(img, in_min=0, in_max=255, out_min=0, out_max=255):
    # Ajusta contraste da imagem de forma linear
    imgf = img.astype(np.float32)
    a = (out_max - out_min) / (in_max - in_min + 1e-8)
    out = (imgf - in_min) * a + out_min
    return clamp(out)

def radiometric_compress_expand(img, gamma=0.5):
    # Compressão e expansão usando potência
    imgf = img.astype(np.float32) / 255.0
    compressed = np.power(imgf, gamma)
    expanded = np.power(compressed, 1.0/gamma)
    return clamp(expanded * 255.0)

def radiometric_sawtooth(img, period=64):
    # Aplica efeito "dente de serra"
    out = img.copy().astype(np.int32)
    out = (out % period) * (255.0 / (period-1))
    return clamp(out)

def radiometric_log(img):
    # Aplica transformação logarítmica
    imgf = img.astype(np.float32)
    c = 255.0 / np.log(1 + 255.0)
    out = c * np.log(1 + imgf)
    return clamp(out)

# ---------------------- 6. Filtros e ruído ----------------------

def add_salt_and_pepper(img, amount=0.1):
    # Adiciona ruído sal e pimenta
    out = img.copy()
    h,w = out.shape[:2]
    num = int(amount * h * w)
    coords = [np.random.randint(0, i, num) for i in (h,w)]
    out[coords[0], coords[1]] = [255,255,255]
    coords = [np.random.randint(0, i, num) for i in (h,w)]
    out[coords[0], coords[1]] = [0,0,0]
    return out

def mean_filter(img, ksize=3):
    # Filtro da média
    pad = ksize//2
    padded = np.pad(img, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            wnd = padded[i:i+ksize, j:j+ksize]
            out[i,j] = wnd.reshape(-1,3).mean(axis=0)
    return clamp(out)

def k_nearest_filter(img, k=5, ksize=5):
    # Filtro k-vizinhos (pega os k mais próximos do pixel central)
    pad = ksize//2
    padded = np.pad(img, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            wnd = padded[i:i+ksize, j:j+ksize]
            center = padded[i+pad, j+pad]
            vec = wnd.reshape(-1,3)
            dists = np.linalg.norm(vec - center, axis=1)
            idx = np.argsort(dists)[:k]
            out[i,j] = vec[idx].mean(axis=0)
    return clamp(out)

def median_filter(img, ksize=3):
    # Filtro da mediana
    pad = ksize//2
    padded = np.pad(img, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            wnd = padded[i:i+ksize, j:j+ksize]
            out[i,j] = np.median(wnd.reshape(-1,3), axis=0)
    return out

def mode_filter(img, ksize=3):
    # Filtro da moda (pega valor mais comum)
    pad = ksize//2
    padded = np.pad(img, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            wnd = padded[i:i+ksize, j:j+ksize]
            vals = wnd.reshape(-1,3)
            pix = []
            for ch in range(3):
                u, cnt = np.unique(vals[:,ch], return_counts=True)
                pix.append(u[np.argmax(cnt)])
            out[i,j] = pix
    return out

# ---------------------- 7. Bordas ----------------------

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def edges_sobel(img, thresh=100):
    # Detecta bordas com Sobel
    g = to_gray(img).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    edges = (mag > thresh).astype(np.uint8) * 255
    return edges

def edges_canny(img, low=100, high=200):
    # Detecta bordas com Canny
    g = to_gray(img)
    return cv2.Canny(g, low, high)

def edges_to_colored_images(img, edges, color=(0,0,255)):
    # Pinta bordas em branco e também na imagem original
    h,w = edges.shape
    white_bg = np.ones((h,w,3), dtype=np.uint8) * 255
    mask = edges.astype(bool)
    white_bg[mask] = color
    colored_on_orig = img.copy()
    colored_on_orig[mask] = color
    return white_bg, colored_on_orig

# ---------------------- 8. BIC ----------------------

def quantize_image(img, n_colors=64):
    # Quantiza cores da imagem
    bins_per_channel = int(round(n_colors ** (1/3)))
    bins_per_channel = max(2, min(256, bins_per_channel))
    step = 256 // bins_per_channel
    q = (img // step) * step + step//2
    return q, bins_per_channel**3

def bic_descriptor(img, quant_img):
    # Calcula histograma de borda e interior
    h,w = quant_img.shape[:2]
    q_flat = quant_img.reshape(-1,3)
    labels = np.array([tuple(p) for p in q_flat])
    labels2d = labels.reshape(h,w)
    border_mask = np.zeros((h,w), dtype=bool)
    for i in range(h):
        for j in range(w):
            val = labels2d[i,j]
            neighs = []
            if i>0: neighs.append(labels2d[i-1,j])
            if i<h-1: neighs.append(labels2d[i+1,j])
            if j>0: neighs.append(labels2d[i,j-1])
            if j<w-1: neighs.append(labels2d[i,j+1])
            for n in neighs:
                if not np.array_equal(val, n):
                    border_mask[i,j] = True
                    break
    uniq, inv = np.unique(q_flat, axis=0, return_inverse=True)
    ncolors = uniq.shape[0]
    border_hist = np.zeros(ncolors, dtype=int)
    interior_hist = np.zeros(ncolors, dtype=int)
    idx2d = inv.reshape(h,w)
    for i in range(h):
        for j in range(w):
            idx = idx2d[i,j]
            if border_mask[i,j]:
                border_hist[idx] += 1
            else:
                interior_hist[idx] += 1
    return border_hist, interior_hist, border_mask

def bic_images_from_mask(orig_img, mask_border):
    # Cria imagens só com borda e só com interior
    h,w = mask_border.shape
    border_img = np.ones((h,w,3), dtype=np.uint8) * 255
    interior_img = np.ones((h,w,3), dtype=np.uint8) * 255
    border_img[mask_border] = orig_img[mask_border]
    interior_img[~mask_border] = orig_img[~mask_border]
    return border_img, interior_img

# ---------------------- Função principal ----------------------

def process_all(input_path, base_outdir, quant_colors=256):
    """Processa uma imagem e salva todos os resultados organizados"""
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {input_path}")

    base = os.path.splitext(os.path.basename(input_path))[0]
    print(f"Processando: {base}")
    
    # 1. BRILHO
    bright_dir = os.path.join(base_outdir, "01_brilho")
    ensure_dir(bright_dir)
    bright = change_brightness(img, 40)
    save_image(os.path.join(bright_dir, f"{base}_brilho_+40.png"), bright)
    bright_neg = change_brightness(img, -40)
    save_image(os.path.join(bright_dir, f"{base}_brilho_-40.png"), bright_neg)

    # 2. NEGATIVO
    neg_dir = os.path.join(base_outdir, "02_negativo")
    ensure_dir(neg_dir)
    neg = negative(img)
    save_image(os.path.join(neg_dir, f"{base}_negativo.png"), neg)

    # 3. HISTOGRAMA GLOBAL
    hist_global_dir = os.path.join(base_outdir, "03_histograma_global")
    ensure_dir(hist_global_dir)
    hg = histogram_global(img)
    np.savetxt(os.path.join(hist_global_dir, f"{base}_histograma_global.txt"), hg, fmt='%d')

    # 4. HISTOGRAMA LOCAL
    hist_local_dir = os.path.join(base_outdir, "04_histograma_local")
    ensure_dir(hist_local_dir)
    hl = histogram_local(img, partitions=(2,3))
    np.savetxt(os.path.join(hist_local_dir, f"{base}_histograma_local.txt"), hl, fmt='%d')

    # 5. TRANSFORMAÇÕES RADIOMÉTRICAS
    radiometric_dir = os.path.join(base_outdir, "05_transformacoes_radiometricas")
    ensure_dir(radiometric_dir)
    
    # Linear
    linear = radiometric_linear_expand(img, 50, 200, 0, 255)
    save_image(os.path.join(radiometric_dir, f"{base}_linear_contraste.png"), linear)
    
    # Gamma
    gamma = radiometric_compress_expand(img, gamma=0.5)
    save_image(os.path.join(radiometric_dir, f"{base}_gamma_0.5.png"), gamma)
    
    # Dente de serra
    sawtooth = radiometric_sawtooth(img, period=64)
    save_image(os.path.join(radiometric_dir, f"{base}_dente_serra.png"), sawtooth)
    
    # Logarítmica
    log_img = radiometric_log(img)
    save_image(os.path.join(radiometric_dir, f"{base}_logaritmica.png"), log_img)

    # 6. FILTROS E RUÍDO
    filters_dir = os.path.join(base_outdir, "06_filtros_e_ruido")
    ensure_dir(filters_dir)
    
    # Adiciona ruído
    noisy = add_salt_and_pepper(img, amount=0.1)
    save_image(os.path.join(filters_dir, f"{base}_ruido_sal_pimenta.png"), noisy)
    
    # Filtros na imagem com ruído
    mean_filtered = mean_filter(noisy, ksize=3)
    save_image(os.path.join(filters_dir, f"{base}_filtro_media.png"), mean_filtered)
    
    k_nearest_filtered = k_nearest_filter(noisy, k=5, ksize=5)
    save_image(os.path.join(filters_dir, f"{base}_filtro_k_vizinhos.png"), k_nearest_filtered)
    
    median_filtered = median_filter(noisy, ksize=3)
    save_image(os.path.join(filters_dir, f"{base}_filtro_mediana.png"), median_filtered)
    
    mode_filtered = mode_filter(noisy, ksize=3)
    save_image(os.path.join(filters_dir, f"{base}_filtro_moda.png"), mode_filtered)

    # 7. DETECÇÃO DE BORDAS
    edges_dir = os.path.join(base_outdir, "07_deteccao_bordas")
    ensure_dir(edges_dir)
    
    # Sobel
    edges_sobel_img = edges_sobel(img, thresh=100)
    sobel_white, sobel_colored = edges_to_colored_images(img, edges_sobel_img)
    save_image(os.path.join(edges_dir, f"{base}_sobel_branco.png"), sobel_white)
    save_image(os.path.join(edges_dir, f"{base}_sobel_colorido.png"), sobel_colored)
    
    # Canny
    edges_canny_img = edges_canny(img, low=100, high=200)
    canny_white, canny_colored = edges_to_colored_images(img, edges_canny_img)
    save_image(os.path.join(edges_dir, f"{base}_canny_branco.png"), canny_white)
    save_image(os.path.join(edges_dir, f"{base}_canny_colorido.png"), canny_colored)

    # 8. DESCRITOR BIC
    bic_dir = os.path.join(base_outdir, "08_descritor_bic")
    ensure_dir(bic_dir)
    
    # Quantiza imagem
    quant_img, n_colors = quantize_image(img, quant_colors)
    save_image(os.path.join(bic_dir, f"{base}_quantizada_{n_colors}cores.png"), quant_img)
    
    # Calcula BIC
    border_hist, interior_hist, border_mask = bic_descriptor(img, quant_img)
    
    # Salva histogramas
    np.savetxt(os.path.join(bic_dir, f"{base}_bic_bordas.txt"), border_hist, fmt='%d')
    np.savetxt(os.path.join(bic_dir, f"{base}_bic_interior.txt"), interior_hist, fmt='%d')
    
    # Cria imagens separadas
    border_img, interior_img = bic_images_from_mask(img, border_mask)
    save_image(os.path.join(bic_dir, f"{base}_bic_apenas_bordas.png"), border_img)
    save_image(os.path.join(bic_dir, f"{base}_bic_apenas_interior.png"), interior_img)

    print(f"{base} processada com sucesso!")

# ---------------------- Função principal ----------------------

def main():
    # Função principal que processa todas as imagens encontradas
    print("=== Processador de Imagens RGB ===")
    print("Buscando imagens na pasta imagens_raw...")
    
    # Busca imagens na pasta imagens_raw
    images = find_images_in_raw()
    
    if not images:
        print("Nenhuma imagem encontrada na pasta imagens_raw!")
        print("Formatos suportados: JPG, JPEG, PNG, BMP, TIFF, TIF")
        return
    
    print(f"Encontradas {len(images)} imagem(ns):")
    for img in images:
        print(f"   - {img}")
    
    # Cria pasta de resultados
    resultados_dir = "resultados"
    ensure_dir(resultados_dir)
    
    # Processa cada imagem
    for img_path in images:
        try:
            process_all(img_path, resultados_dir, quant_colors=256)
        except Exception as e:
            print(f"Erro ao processar {img_path}: {e}")
    
    print(f"\nProcessamento concluído!")
    print(f"Resultados salvos em: {resultados_dir}/")
    print("\nEstrutura de pastas:")
    print("  resultados/")
    print("  ├── 01_brilho/")
    print("  ├── 02_negativo/")
    print("  ├── 03_histograma_global/")
    print("  ├── 04_histograma_local/")
    print("  ├── 05_transformacoes_radiometricas/")
    print("  ├── 06_filtros_e_ruido/")
    print("  ├── 07_deteccao_bordas/")
    print("  └── 08_descritor_bic/")

if __name__ == '__main__':
    main()
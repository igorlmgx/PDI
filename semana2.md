[voltar](index.md/)
## Histograma

#### Sinal de 1 dimensão

Python
```python
def histograma1d(signal, num_bins):

    v_min = min(signal)
    bin_size = (max(signal)-v_min+1)/num_bins
    hist = np.zeros(num_bins)
    num_values = len(signal)
    for value in signal:
        bin_index = (value-v_min)/bin_size
        bin_index = int(bin_index)
        hist[bin_index] += 1
    
    bins_values = np.zeros(num_bins+1)
    for i in range(num_bins+1):
        bins_values[i] = v_min + i*bin_size

    return (bins_values, hist, bin_size)
```

Numpy
```python
hist, bins = np.histogram(signal, num_bins)
```

#### Imagem

Python
```python
def histograma2d(img, num_bins):
    bin_size = 256/num_bins

    hist = np.zeros(num_bins)
    num_rows, num_cols = img.shape
    for row in range(num_rows):
        for col in range(num_cols):
            bin_index = int(img[row, col]/bin_size)
            hist[bin_index] += 1
    
    bins_values = np.linspace(0, 256, num_bins)

    return (bins_values, hist, bin_size)
```

Matplotlib
```python
plt.hist(img.flatten(), num_bins, edgecolor='k')
```

#### Equalização de histograma

```python
def histogram_equalization(img):
    bins = range(0, 257)
    hist, _ = np.histogram(img, bins)

    mn = sum(hist)
    c = 255./mn
    out_intensity = np.zeros(256)
    for k in range(256):
        soma = 0
        for j in range(0, k+1):
            soma += hist[j]
        out_intensity[k] = c*soma
    
    img_eq = np.zeros(img.shape)
    num_rows, num_cols = img.shape
    for row in range(num_rows):
        for col in range(num_cols):
            img_eq[row, col] = out_intensity[img[row, col]]
    
    return img_eq
```

## Correlação-cruzada e Convolução

#### Correlação-cruzada

Python
```python
def correlation(img, w):
    num_rows, num_cols = img.shape
    f_num_rows, f_num_cols = w.shape

    half_r_size = f_num_rows//2
    half_c_size = f_num_cols//2
    
    img_padded = np.zeros((num_rows+f_num_rows-1, num_cols+f_num_cols-1), dtype=img.dtype)
    for row in range(num_rows):
        for col in range(num_cols):   
            img_padded[row+half_r_size, col+half_c_size] = img[row, col]
    
    img_filtered = np.zeros((num_rows, num_cols))
    for row in range(num_rows):
        for col in range(num_cols):
            sum_region = 0
            for s in range(f_num_rows):
                for t in range(f_num_cols):
                    sum_region += w[s, t]*img_padded[row+s, col+t]
            img_filtered[row, col] = sum_region
            
    return img_filtered
```

Otimizações com Numpy
```python
def correlation(img, w):
    num_rows, num_cols = img.shape
    f_num_rows, f_num_cols = w.shape

    half_r_size = f_num_rows//2
    half_c_size = f_num_cols//2
    
    img_padded = np.pad(img, ((half_r_size,half_r_size),(half_c_size,half_c_size)), mode='constant')
    
    img_filtered = np.zeros((num_rows, num_cols))
    for row in range(num_rows):
        for col in range(num_cols):
            patch = img_padded[row:row+f_num_rows, col:col+f_num_cols]
            img_filtered[row, col] = np.sum(w*patch)
            
    return img_filtered
```

Scipy
```python
scipy.signal.correlate(img, w, mode='same')
```

#### Convolução

Correlação-cruzada com filtro invertido
```python
def convolution(img, w):
    w_inv = w[::-1, ::-1]

    return correlation(img, w_inv)
```

Scipy
```python
scipy.signal.convolve(img, w, mode='same')
```

## Suavização

#### Média simples

```python
tam = 5
w = np.full([tam,tam], 1./tam**2)
```
#### Suavização gaussiana

Filtro 1D
```python
def gaussian_filter_1d(filter_size):
    sigma = filter_size/6.
    x = np.linspace(-3*sigma, 3*sigma, filter_size)
    y = np.exp(-x**2/(2*sigma**2))
    
    y = y/np.sum(y)

    return y
```

Filtro 2D
```python
def gaussian_filter_2d(filter_size):
    sigma = filter_size/6.
    x_vals = np.linspace(-3*sigma, 3*sigma, filter_size)
    y_vals = x_vals.copy()
    z = np.zeros((filter_size, filter_size))
    for row in range(filter_size):
        x = x_vals[row]
        for col in range(filter_size):
            y = y_vals[col]
            z[row, col] = np.exp(-(x**2+y**2)/(2*sigma**2))
    z = z/np.sum(z)

    return z
```

#### Função seno com ruído

```python
def corrupt_sin(S, r):
    x = np.linspace(0, 4*np.pi, S)
    return np.sin(x) + (r * np.random.rand(len(x)))
```

#### Exemplo pacman

```python
pacman = plt.imread('pacman.tiff')
pacman_thresholded = pacman > 110

w = gausian_filter_2d(9)
pacman_filtered = scipy.signal.convolve(pacman, w, mode='same')

pacman_filt_thresholded = pacman_filtered>110
plt.imshow(pacman_filt_thresholded, 'gray')
```
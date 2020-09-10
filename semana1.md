[voltar](index.md/)
# Conceitos básicos

### Formas de imprimir variáveis

```python
print('valor = %d'%valor)
print('valor = {}'.format(valor))
print(f'valor = {valor}')
```

### Estrutura condicional

```python
if cond1 and cond2:
    print("primeiro if")
elif cond3 and cond 4:
    print("usando elif")
else:
    print("usando else")
```

### Estrutura de repetição

```python
for val in lista:
    print(val)

for i in range(len(lista)):
    print(lista[i])

for i in range(10):
    print(1)
```

### Listas

```python
lista = [1, 2, True, "quatro", 5]

matriz = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

lista.append("seis")

valor = lista.pop()

print(lista[0:3])
```

### Funções

```python
def soma(v1, v2):
    return v1 + v2

print(soma(3, 4))
```

# Numpy

### Vetores

```python
vetor = [2.3, 5.4, 3.3, 2.5, 7.3]

v = np.array(vetor)

mult = 5 * v
soma = v + v
prodEscalar = sum(v * v)

media = np.mean(v)
desvioPadrao = np.std(v)
```

### Matrizes

```python
matriz = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

determinante = np.linalg.det(matriz)
produtoMatricial = np.dot(matriz, matriz)
soma = matriz + matriz
```

### Sistemas de equações

```python
# 3x + 5y  + 3z = 54.6
# x  + 12y + 5z = 98.8
# 5x + 7y  + 2z = 65.2

coeficientes = [[3,  5, 3],
                [1, 12, 5],
                [5,  7, 2]]
valores = [54.6, 98.8, 65.2]

res = np.linalg.solve(coeficientes, valores)

print("x={:.2}, y={:.2}, z={:.2}".format(res[0], res[1], res[2]))
```

# Imagens

### Leitura e exibição

```python
img = plt.imread('earth_gray.jpg')
plt.imshow(img, cmap='gray')
```

### Imagens coloridas e canais

```python
img = plt.imread('earth.jpg')

img_red = img[:,:,0]
img_green = img[:,:,1]
img_blue = img[:,:,2]

plt.figure(figsize=[15, 5.])
plt.subplot(1, 3, 1)
plt.imshow(img_red, cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(img_green, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(img_blue, cmap='gray')
```

# Transformações pontuais

### Negativo

Em python
```python
num_linhas, num_cols = img.shape
img_neg = np.zeros((num_linhas, num_cols), dtype=np.uint8)
for row in range(num_linhas):
    for col in range(num_cols):
        img_neg[row, col] = 255 - img[row, col]
plt.imshow(img_neg, 'gray')
```

Com o maravilhoso numpy
```python
img_neg_np = 255 - img
plt.imshow(img_neg_np, 'gray')
```

### Logaritmo

Python
```python
img_log = np.zeros((num_linhas, num_cols), dtype=np.float)
for row in range(num_linhas):
    for col in range(num_cols):
        img_log[row, col] = np.log(1+img[row, col])
plt.imshow(img_log, 'gray')
```

Numpy
```python
img_log = np.log(img.astype(float))
plt.imshow(img_log, 'gray')
```

Lookup table
```python
for value in range(0, 256):
    lookupTable[value] = np.log(1+value)

img_log = np.zeros((num_linhas, num_cols), dtype=np.float)
for row in range(num_linhas):
    for col in range(num_cols):
        img_log[row, col] = lookupTable[img[row, col]]
plt.imshow(img_log, 'gray')
```

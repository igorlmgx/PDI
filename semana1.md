[voltar](index.md/)
# Semana 1
## Conceitos básicos

Diferentes formas de imprimir variáveis no terminal

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
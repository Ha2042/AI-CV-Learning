# 01 Python

## 2026-9-27

## Today I reviewed:

- List
- Tuple
- dictionary
- function
- class
- import

## Problems

函数可以作为变量传递

def add(a,b):

    return a+b
    
f=add

f(2,3)相当于add(2,3)

def preprocess(images):
    processed = []

    for image in images:
        image = image / 255.0
        image = image.reshape(28, 28)
        processed.append(image)

    return processed


def predict(model, images):
    images = preprocess(images)

    results = []

    for image in images:
        output = model(image)
        results.append(output)

    return results
    
以上代码中的output=model(image)就是model作为变量传递
## Next

Learn NumPy arry and matrix operations.

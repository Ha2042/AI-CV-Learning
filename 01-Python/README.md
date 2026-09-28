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

 ### model作为函数predict的变量，为什么还可以写成output = model(image)?   

 ### 在Python语法中，函数可以作为变量传递。这里的 model 不是“必须是模型对象”或“不能调用”，它只是变量名。只要传进来的对象是可调用的，就能写成 model(image)。
 
    def f(x):
        return x * 2
    
    def g(x):
        return x + 10
    
    def predict(model, images):
        return [model(img) for img in images]
    
    print(predict(f, [1, 2, 3]))  # [2, 4, 6]
    print(predict(g, [1, 2, 3]))  # [11, 12, 13]
    
    model(image) 到底等于谁，取决于调用时传入了什么
    
## Next

Learn NumPy arry and matrix operations.

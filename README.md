

<center>

![SVM diagram](https://miro.medium.com/v2/resize:fit:828/format:webp/0*8QauwS7cvn7mCbH0.png)


</center>

# **What is SVM (Support Vector Machine)?**

Support Vector Machine (SVM) is a **powerful supervised learning algorithm** used for **classification** and **regression**. It works by finding the best / optimal **decision boundary (hyperplane)** that separates different classes in a dataset.

SVM is widely used in:

🖼️ **Image classification** (e.g., digit recognition, facial recognition, medical imaging, object detection, handwriting)

💬 **Text classification** (e.g., spam detection, sentiment analysis)

🧬 **Bioinformatics** (e.g., disease classification from genetic data, gene classification)

🕵️ **Anomaly Detection** (e.g., fraud detection, cybersecurity)

**Goal of SVM**: Find a hyperplane that best separates different classes with **the maximum margin**.

**When to use SVM?**
- When the dataset is small to medium-sized
- When high-dimensional data is involved
- When computational efficiency is a concern (compared to deep learning)


### **What are Hyperplanes, Margins and Support Vectors?**

✈️ **Hyperplane**
- A **hyperplane** is a boundary that separates different classes.
- In **2D space**, it's a **line**.
- In **3D space**, it's a **plane**.
- In **higher dimensions**, it's a more complex shape.

🚀 **Margin**
- The **margin** is the distance between the hyperplane and the closest data points (support vectors).
- SVM tried to **maximize** this margin to improve classification performance.

🔍 **Support Vectors**
- The **support vectors** are the data points that are closest to the decision boundary.
- These points **determine** the position of the hyperplane.
- If these points change, the hyperphane changes too!

$\Rightarrow$ The SVM algorithm finds the best **straight line** that separates them **while keeping the margin as large as possible**.

🚀 **Why maximize the margin?**
- Large margins = better generalization (good on new data)
- Small margin = overfitting (too specific to training data)

### **Hard Margin vs Soft Margin SVM**
There are 2 types of SVM based on how strict we want the separation to be:

👍 **Hard Margin SVM**
- **Strictly** separates classes with a hard boundary
- NO misclassified points are allowed
- **Works well for perfectly separable data** 
- **Problem?** It **fails** if the data was **overlapping classes** or **noise**

👍 **Soft Margin SVM**
- Allows **some misclassifications**
- Uses a parameter **C** to control the trade-off between **margin size** and **misclassification**
- **Better for real-world data**, where perfect separation is not possible $\Rightarrow$ It balances model flexibility and accuracy

# **What is SVC (Support Vector Classifier)?**

SVC (**Support Vector Classifier**) is the classification version of SVM. It supports different **kernel functions** and allows hyperparameter tuning using **GridSearchCV** to optimize **C and gamma** for better accuracy.

- It uses **SVM principles** to classify data points into different categories
- It can handle **both linear and non-linear classification**
- **`sklearn.svm.SVC()`** is used to implement it in Python

### **List of SVC Parameters**

**Kernel**

<center>

| Parameters | Default | Description |
| ----- | ----- | ----- |
| **`kernel`** | **`'rbf'`** | Specifies the kernel type (**`linear`**, **`poly`**, **`rbf`**, **`sigmoid`**, **`precomputed`**) | 
| **`degree`** | 3 | Degree if polynomial kernel (**`poly`** kernel ONLY). Ignored for other kernels | 
| **`gamma`** | **`scale`** | Controls how far the influence of a training example reaches (**`scale`**, **`auto`** or a float value) | 
| **`coef0`** | 0.0 | Independent term in **`poly`** and **`sigmoid`** kernels |

</center>

**Regularization & Decision boundaries**

| Parameter | Default | Description |
| ----- | ----- | ----- |
| **`C`** | 1.0 | Regularization parameter (higher = stricter margin, lower = softer margin) | 
| **`shrinking`** | **`True`** | Whether to use the shrinking heuristic for faster convergence | 

**Probability & Class weights**

| Parameter | Default | Description |
| ----- | ----- | ----- |
| **`probability`** | **`False`** | Whether to enable probability estimates (makes prediction shower if **`True`**) | 
| **`class_weight`** | **`None`** | Adjusts weights for imbalanced classes (**`balanced`** or a dictionary of class weights)

**Computational performance**

| Parameter | Default | Description |
| ----- | ----- | ----- |
| **`tol`** | 1e-3 | Tolerance for the stopping criterion | 
| **`cache_size`** | 200 | Size of cache (in MB) for storing kernel matrix computations | 
| **`max_iter`** | -1 | Maximum iterations (**-1** means no limit) | 



**Multi-class & Miscellaneous**

| Parameter | Default | Description |
| ----- | ----- | ----- |
| **`decision_function_shape`** | **`ovr`** | **`ovr`** (one-vs-rest) or **`ovo`** (one-vs-one) for multi-class classification | 
| **`break_ties`** | **`False`** | Whether to break ties when making predictions (**`True`** only works with **`decision_function_shape='ovr'`**) | 
| **`random_state`** | **`None`** | Controls randomness when using probability estimates |

### **Commonly Tuned Hyperparameters**

📌 **`kernel`** - defines how data is transformed

SVC supports different **kernel functions** that transform data into a higher-dimensional space to make it separable.

1️⃣ **Linear Kernel** (**`kernel='linear'`**)
- Best for linearly separable data
- Find a straight-line decision boundary
- Example: Spam vs. Non-spam emails

2️⃣ **Polynomial Kernel** (**`kernel='poly'`**)
- Maps features using polynomial transformations
- Useful for data with polynomial relationships
- Example: Handwriting recognition

📌 degree – controls polynomial kernel complexity

- **Higher `degree`** → More **complex** decision boundary
- **Lower `degree`** → **Simpler** decision boundary
- **Default is 3**, but it can be tuned based on data

3️⃣ **Radial Basis Function (RBF) Kernel** (**`kernel='rbf'`**)
- Best for **non-linear** data
- Transforms data into higher dimensions
- Use **Gaussian functions** to separate complex patterns
- Example: Facial recognition, medical diagnosis

4️⃣ **Sigmoid Kernel** (**`kernel='sigmoid'`**)
- Similar to neural networks
- Less commonly used

📌 **`C`** – regularization parameter

**Controls** how much SVC **allows misclassification**

- **High `C`** (e.g., 1000) $\rightarrow$ less margin, fewer misclassifications (risk of overfitting)
- **Low `C`** (e.g., 1) $\rightarrow$ more margin, allow misclassification (better generalization)

🔎 Think of **`C`** like this:

If **`C`** is **very high**, the model **focuses on classifying every point correctly**, even if the decision boundary is complicated (risk of overfitting).

If **`C`** is **low**, the model **allows some misclassifications**, creating a smoother boundary (better generalization).

📌 **`gamma`** – controls the influence of points in **`RBF`** kernel

Only used in **RBF** and **polynomial** kernels

- High **gamma** $\rightarrow$ points have a **narrow** influence, creating **complex** boundaries (risk of overthinking)
- Low **gamma** $\rightarrow$ points have a **wider** influence, creating a **smoother** boundary (better generalization)

📌 How to tune **`gamma`**?

Start with **`auto`** or **`scale`** and then fine-tune using Grid Search or Random Search.


**Note**: Considering **`probability`** and **`class_weigth`**

### **How SVM works?**
1. It **maps** the input data into **a higher-dimensional space** using kernels (e.g., **`linear`**, **`RBF`**).
2. It finds **the optimal hyperplane** that **maximizes the margin** between classes.
3. Some **misclassifications are allowed** based on **`C`** to balance flexibility and accuracy.
4. If the data is **not linearly separable**, **`kernels`** like **`RBF`** transform it into a higher-dimensional space where it is separable.

### **Conclusion**
SVM remains a powerful tool in machine learning due to its ability to generalize well, handle high-dimensional data, and provide robust classification boundaries. However, for extremely large datasets or overlapping classes, deep learning or ensemble methods may perform better.

Read more: <a href='https://serokell.io/blog/support-vector-machine-algorithm'>Support Vector Machine Algorithm</a>
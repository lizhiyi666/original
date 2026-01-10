设计可微约束函数 $g(\cdot)$ 需要遵循“可导”和“惩罚违规”两个原则。

以下是具体的推导步骤和设计方案：

### 1. 问题定义

假设有 $K$ 对约束关系：
$\mathcal{S} = \{ (A_1, B_1), (A_2, B_2), \dots, (A_K, B_K) \}$
其中每一对 $(A_k, B_k)$ 都要求：**序列中** **$B_k$** **不可出现在** **$A_k$** **之前。**

* **序列位置 (Sequence Position, index** **$i$)**

  * 范围：$i \in \{1, \dots, L\}$。

* **扩散时间步 (Diffusion Timestep, index** **$t$)**

  * 范围：$t \in \{T, T-1, \dots, 0\}$。

### 2. 中间变量计算：类别概率投影

假设我们处于 **扩散时间步** **$t$**（比如第 50 步去噪）：

* 模型看到的是一个完整的矩阵 $\boldsymbol{Y}_t \in \mathbb{R}^{L \times N}$。

* 这个矩阵包含了序列中 **所有位置** **$i \in \{1, \dots, L\}$** 的概率分布。

我们计算约束 $g(\cdot)$ 时，是在**同一个扩散时间步** **$t$** 内，对 **不同序列位置** **$i$** **和** **$j$** 进行计算。

* **$t$**：代表扩散的去噪进度。

* **$i, j$**：代表序列中的位置索引。

**在扩散的第** **$t$** **步（外层）：**
模型输出了当前对整个序列的预测矩阵 $\boldsymbol{Y}_t$（形状为 $L \times N$）。

我们定义位置 $i$ 属于 A 类或 B 类的概率（注意下标是 $i$）：

$$
P_A(i) = \sum_{v \in \mathcal{A}} \boldsymbol{Y}_t[i, v]
$$

$$
P_B(i) = \sum_{v \in \mathcal{B}} \boldsymbol{Y}_t[i, v]
$$

### 3. $g(\cdot)$ 函数设计方案

对于任一约束$(A_k, B_k)$，其逆反命题是：**“严禁** **$B_k$** **出现在$A_k$前面”**。
换句话说，对于序列中任意两个位置 $i$ 和 $j$，如果 $i < j$（$i$ 在前），那么**不能**发生“第 $i$ 个是 $B_k$且 第 $j$ 个是 $A_k$”的情况。

基于此，我们可以设计一个\*\*“逆序对惩罚积 (Pairwise Inversion Penalty)”\*\*作为 $g(\cdot)$。

定义总约束函数：

$$
g_{\text{total}}(\boldsymbol{Y}) = \sum_{k=1}^{K} g_k(\boldsymbol{Y}) = \sum_{k=1}^{K} \left( \sum_{1 \le i < j \le L} P_{B_k}(i) \cdot P_{A_k}(j) \right)
$$

#### 含义：

* 我们遍历所有的时间对 $(i, j)$，其中 $i$ 早于 $j$。

* 如果模型在 $i$ 处倾向于生成 B（$P_{B_k}(i)$ 大），**并且**在后面的 $j$ 处倾向于生成 A（$P_{A_k}(j)$ 大），那么乘积 $P_{B_k}(i) \cdot P_{A_k}(i)$ 就会很大。

* 这就构成了一个“违规能量”。我们将所有可能的逆序违规累加起来。

* 如果序列严格满足 $A_k \dots A_k \to \text{Others} \to B_k \dots B_k$，那么对于任何 $i < j$，要么 $P_{B_k}(i) \approx 0$，要么 $P_{A_k}(j) \approx 0$，总和 $g(\boldsymbol{Y}) \approx 0$。

### 4. 矩阵化实现（为了高效计算）

在 GPU 上实现时，双重循环效率低。这个公式可以写成矩阵形式。

#### 步骤 A：构建概率矩阵

假设当前扩散时间步 $t$ 的模型输出为 $\boldsymbol{Y} \in \mathbb{R}^{L \times N}$。
我们构建两个 **特征概率矩阵** $\mathbf{P}_A, \mathbf{P}_B \in \mathbb{R}^{L \times K}$。

* $\mathbf{P}_A$ 的第 $k$ 列：对应序列在各个位置生成 $A_k$ 类别的概率。

* $\mathbf{P}_B$ 的第 $k$ 列：对应序列在各个位置生成 $B_k$ 类别的概率。

这可以通过简单的 `gather` 或 `index_select` 操作从 $\boldsymbol{Y}$ 中提取。

#### 步骤 B：掩码矩阵 (Mask)

定义上三角矩阵 $M \in \mathbb{R}^{L \times L}$：
$M_{i, j} = \begin{cases} 1 & \text{if } i < j \\ 0 & \text{otherwise} \end{cases}$
这代表了“$i$ 在前，$j$ 在后”的时序关系。

#### 步骤 C：一次性计算 (Einstein Summation)

所有的违规惩罚可以通过以下公式一次算出：

$$
g_{\text{total}} = \text{Sum} \left( (\mathbf{P}_B^\top \cdot M) \odot \mathbf{P}_A^\top \right)
$$

或者用更直观的公式表示：

$$
g_{\text{total}} = \sum_{k=1}^K \sum_{i=1}^L \sum_{j=1}^L \mathbf{P}_B[i, k] \times M[i, j] \times \mathbf{P}_A[j, k]
$$

### 5. 为什么这个 $g(\cdot)$ 是有效的？（梯度分析）

在 CDD 的优化过程中（ALM 算法），优化器会计算 $\nabla_{\boldsymbol{y}} g(\boldsymbol{y})$。我们来看看梯度会如何指导模型：

假设当前生成了一个违规序列：`... B ... A ...` （B 在前，A 在后）。

* **对于前面的 B (位置** **$i$)：** 梯度会告诉模型：“后面有个 A，所以你这里选 B 的概率 $P_B(i)$ 必须降低”。

* **对于后面的 A (位置** **$j$)：** 梯度会告诉模型：“前面已经有个 B 了，所以你这里选 A 的概率 $P_A(j)$ 必须降低”。

这种梯度的双向传导完全符合论文中关于 Differentiable Projection 的要求，能有效地将概率分布推向合规区域（即：前面的变成 A 或其他，或者后面的变成 B 或其他）。

### 6. 阈值设置

* **阈值** **$\tau$：** 设为 $0$（或一个极小的正数 $\epsilon$）。

* **违规计算：** $\Delta g = \max(0, g(\boldsymbol{y}) - 0)$。

### 总结

设计步骤如下：

1. **输入：** 序列概率分布 $\mathbf{Y}$。

2. **聚合：** 将 $\mathbf{Y}$ 聚合为“特征概率矩阵” $\mathbf{P}_A$ 和 $\mathbf{P}_B$。

3. **计算：** 计算所有“前B后A”的概率积之和：

   $$
   g_{\text{total}}(\boldsymbol{Y}) = \sum_{k=1}^{K} g_k(\boldsymbol{Y}) = \sum_{k=1}^{K} \left( \sum_{1 \le i < j \le L} P_{B_k}(i) \cdot P_{A_k}(j) \right)
   $$

4. **求导：** 该函数完全由加法和乘法组成，且输入来自 Gumbel-Softmax，因此完全可微，可直接插入 CDD 的 ALM 循环中。


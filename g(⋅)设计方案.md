

### 1. 总体架构融合 (Architecture Fusion)

*   **基座模型 (Backbone):** 使用 Marionette 的完整架构（图 1）。
    *   保留 **Context Alignment & Extraction** (Sec 4.1)：处理天气、时间等全局/局部上下文。
    *   保留 **Conditional Temporal Modeling** (Sec 4.2)：使用 TPP 生成事件时间戳 $t_1, \dots, t_L$。
    *   保留 **Conditional Spatial Modeling** (Sec 4.3)：这是我们需要植入 CDD 的地方。
*   **控制机制 (Controller):** 使用 CDD 的 **Projected Diffusion Sampling** (Sec 4 of CDD paper)。
    *   替代 Marionette 原有的简单反向采样。

---

### 2. 序列定义与约束适配

根据 Marionette 论文的 **公式 (7)**，空间序列的结构是：
$$ \mathbf{x} = [\langle \text{sos} \rangle, k_1, \dots, k_J, \langle \text{sep} \rangle, l_1, \dots, l_J, \langle \text{eos} \rangle] $$
其中：

*   $k_{1:J}$ 是 **P OI 类别 (Activity Categories)** 序列（这是我们要施加偏序约束的对象）。
*   $l_{1:J}$ 是 **POI 实体** 序列（这部分依赖于类别，且暂时不加约束）。

**约束目标：** 仅针对 $k_1, \dots, k_J$ 这部分子序列，满足偏序约束 $\mathcal{S}$（如“餐饮”必须在“娱乐”前）。

**$g(\cdot)$ 的适配：**
我们需要修改之前的 $g(\cdot)$ 函数，使其能够处理 Marionette 的特殊序列结构。
*   **掩码 (Masking):** 只关注 $k$ 部分，忽略特殊的 `<sos>`, `<sep>`, `<eos>` 以及后面的 POI $l$ 部分。

---

### 3. 详细算法流程 (The Algorithm)

我们将 CDD 的 **ALM 投影循环** 嵌入到 Marionette 的 **Cascading Conditioning Denoising (Sec 4.3.2)** 过程中。

#### 输入
*   上下文条件 $Y$ 。
*   时间序列 $\tau$ (由 TPP 生成)。
*   约束集 $\mathcal{S} = \{(A, B)\}$ 。

#### 反向扩散过程 (Reverse Process)

假设扩散步数为 $N$，从 $n = N$ 到 $1$ 进行迭代：

**Step 1: 预测 (Predict)**
Marionette 的去噪网络（Transformer Decoder）根据当前噪声状态 $\mathbf{x}_n$、时间步 $n$ 和条件 $Y$，预测 **$x_0$ 的 Logits**：
$$ \text{logits}_{\mathbf{x}_0} = \text{Model}(\mathbf{x}_n, n, Y) $$
*   `logits` 形状为 $[1, \text{SeqLen}, \text{VocabSize}]$。

**Step 2: 投影 (Project - The CDD Core)**
这是核心修改点。我们不直接用预测的 logits 采样，而是先优化它。

1.  **切片 (Slicing):** 从 `logits` 中提取出 **Category 部分** 的 Logits。
    *   假设 `<sos>` 是第 0 位，那么 Category 是第 $1$ 到 $J$ 位。
    *   $\mathbf{u}_{cat} = \text{logits}[:, 1:J+1, :]$

2.  **ALM 优化循环 (Inner Loop):**
    令 $\mathbf{y} = \mathbf{u}_{cat}$ (可优化变量)。
    *   **Loop $k=1 \dots K_{inner}$:**
        1.  **Gumbel-Softmax:** $\mathbf{P} = \text{Softmax}((\mathbf{y} + \xi)/T)$。
        2.  **提取特征:** 从 $\mathbf{P}$ 中提取 A 类和 B 类的概率列 $\mathbf{p}_A, \mathbf{p}_B$。
        3.  **计算约束:** $g(\mathbf{P}) = \mathbf{p}_B^T \cdot M_{upper} \cdot \mathbf{p}_A$ (使用前文设计的矩阵乘法)。
        4.  **计算梯度:** $\nabla \mathcal{L}_{ALM}$。
        5.  **更新:** $\mathbf{y} \leftarrow \mathbf{y} - \eta \nabla$。
    *   **End Loop**

3.  **回填 (Scatter):** 将优化后的 $\mathbf{y}$ 放回原 `logits` 的对应位置。

**Step 3: 采样 (Sample)**
使用修正后的 `logits`（代表 $p_\theta(x_0|x_n)$），结合 Marionette 的 Mask-and-Replace 转移矩阵（公式 10, 11），采样出 $\mathbf{x}_{n-1}$。

---

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

---

### 算法：基于 ALM 的 POI 序列偏序约束投影
**Algorithm: Constrained Projection via Augmented Lagrangian Method**

**输入 (Inputs):**
*   原始模型预测 Logits: $\boldsymbol{u}_t \in \mathbb{R}^{B \times L \times V}$ (来自扩散模型)
*   约束映射矩阵: $\mathbf{W}^A, \mathbf{W}^B \in \{0,1\}^{V \times K}$ (预计算，$K$ 为约束对数量)
*   超参数: 学习率 $\eta$, 存在性权重 $w_{exist}$, 初始乘子 $\boldsymbol{\lambda}_0$, 罚参数 $\mu_0$, 增长系数 $\alpha$
*   迭代次数: 外层 $T_{out}$, 内层 $T_{in}$

**输出 (Output):**
*   修正后的 Logits: $\boldsymbol{y}^*$

---

**过程 (Procedure):**

1:  **初始化**:
    $\boldsymbol{y} \leftarrow \boldsymbol{u}_t$  (初始化优化变量为原始预测)
    $\boldsymbol{\lambda} \leftarrow \boldsymbol{\lambda}_0, \quad \mu \leftarrow \mu_0$
    $\mathbf{Q}_{target} \leftarrow \text{Softmax}(\boldsymbol{u}_t)$ (冻结原始分布作为 KL 散度目标)

2:  **For** $step_{out} = 1$ **to** $T_{out}$ **do**:  (外层循环：更新 ALM 参数)

3: 	     **For** $step_{in} = 1$ **to** $T_{in}$ **do**:  (内层循环：梯度下降优化 y)

4:     	   	  **// 1. Gumbel-Softmax 松弛 (可微近似)**
            		$\tilde{\boldsymbol{x}} = \text{Softmax}\left( (\boldsymbol{y} + \boldsymbol{\epsilon}) / \tau \right), \quad \boldsymbol{\epsilon} \sim \text{Gumbel}(0,1)$

5:          		**// 2. 约束违规计算 (矩阵化高效计算)**
            		(提取约束涉及类别的概率矩阵, $\in \mathbb{R}^{B \times L \times K}$)
            		$\mathbf{P}^A = \tilde{\boldsymbol{x}} \cdot \mathbf{W}^A$
            		$\mathbf{P}^B = \tilde{\boldsymbol{x}} \cdot \mathbf{W}^B$

​					(A. 顺序违规 $g_{order}$: 利用前缀和计算 $B$ 在 $A$ 之前的概率积)
​            		$\mathbf{S}^B[t] = \sum_{i=0}^{t-1} \mathbf{P}^B[i] \quad$ *(Prefix Sum, $O(L)$)*
​            		$g_{order} = \sum_{t=1}^{L} \sum_{k=1}^{K} \left( \mathbf{S}^B[t, k] \cdot \mathbf{P}^A[t, k] \right)$

​					(B. 存在性违规 $g_{exist}$: 强制 A/B 期望出现次数 $\ge 1$)
​            		$C^A = \sum_{t=1}^{L} \mathbf{P}^A[t], \quad C^B = \sum_{t=1}^{L} \mathbf{P}^B[t]$
​            		$g_{exist} = \sum_{k=1}^{K} \left( \text{ReLU}(1 - C^A_k) + \text{ReLU}(1 - C^B_k) \right)$

​					(C. 总软违规向量 $\in \mathbb{R}^B$)

​					$\boldsymbol{g}_{soft} = g_{order} + w_{exist} \cdot g_{exist}$

6:         	 	**// 3. 计算损失函数**
            		$\mathcal{L}_{KL} = \frac{1}{B} \sum_{i=1}^{B} D_{KL}(\mathbf{Q}_{target}^{(i)} \| \text{Softmax}(\boldsymbol{y}^{(i)}))$

​					$\mathcal{L}_{constraint} = \frac{1}{B} \sum_{i=1}^{B} \left( \boldsymbol{\lambda}^{(i)} \cdot \boldsymbol{g}_{soft}^{(i)} + \frac{\mu}{2} (\boldsymbol{g}_{soft}^{(i)})^2 \right)$	

​					$\mathcal{L}_{total} = \mathcal{L}_{KL} + \mathcal{L}_{constraint}$

7:         	 	**// 4. 梯度更新**
            		$\boldsymbol{y} \leftarrow \boldsymbol{y} - \eta \cdot \nabla_{\boldsymbol{y}} \mathcal{L}_{total}$    

​            		**End Inner Loop**

9:      	**// 5. 外层更新 (使用硬判定 Hard Validation)**
        	$\boldsymbol{x}_{hard} = \text{OneHot}(\text{argmax}(\boldsymbol{y}))$
        	计算硬违规 $\boldsymbol{g}_{hard}$ (计算逻辑同步骤 5，但输入为离散的 $\boldsymbol{x}_{hard}$)
        	$\boldsymbol{\lambda} \leftarrow \boldsymbol{\lambda} + \mu \cdot \boldsymbol{g}_{hard}$

​			$\mu \leftarrow \min(\alpha \cdot \mu, \mu_{max})$

10: **End Outer Loop**

11: **Return** $\boldsymbol{y}$

---

### 关键计算公式详解

#### 1. 顺序违规 (高效前缀和)
利用前缀和代替矩阵乘法，计算所有满足 $i < j$ 且 $Sequence[i]=B, Sequence[j]=A$ 的概率总和：
$$ g_{order}^{(k)} = \sum_{j=1}^{L} P^A_{j,k} \cdot \underbrace{\left( \sum_{i=1}^{j-1} P^B_{i,k} \right)}_{\text{Prefix Sum of } B} $$

#### 2. 存在性违规 (ReLu 铰链损失)
确保类别 $A$ 和 $B$ 在序列中的概率总和至少为 1：
$$ g_{exist}^{(k)} = \max\left(0, 1 - \sum_{t=1}^{L} P^A_{t,k}\right) + \max\left(0, 1 - \sum_{t=1}^{L} P^B_{t,k}\right) $$

#### 3. 增广拉格朗日损失 (ALM Objective)
这是内层优化的核心目标函数：
$$ \mathcal{L}(\boldsymbol{y}) = \underbrace{D_{KL}(\boldsymbol{u}_t \| \boldsymbol{y})}_{\text{Fidelity}} + \underbrace{\boldsymbol{\lambda} g(\boldsymbol{y})}_{\text{Adaptive Penalty}} + \underbrace{\frac{\mu}{2} \|g(\boldsymbol{y})\|^2}_{\text{Augmented Penalty}} $$

分析其余指标下降的原因，是因为投影不可避免还是可以优化

多找些baseline



## 评估指标计算方法

---

## 1. **Distance**

**含义:**  
统计每一条轨迹的总移动距离（travel distance）。  

**计算方法:**  
- 对于每条轨迹的GPS点序列，用 haversine 公式 (`distance` 函数) 计算相邻两点的地理距离。
- 用 `np.sum` 求和，得到整条轨迹的总距离 (`travel_distance` 函数)。
- 对所有轨迹分别计算，形成生成数据与真实数据的 `Distance` 分布，最后用 `evaluation` 计算JSD。

---

## 2. **Radius**  
**含义:**  
每条轨迹点到中心点（均值点）的均方根距离（描述轨迹的空间扩散程度）。  

**计算方法:**  
- 用 `np.mean` 计算所有 GPS 点的中心坐标。
- 用 `distance` 计算每个点到中心点的距离，取均值再开平方 (`radius` 函数)。
- 分类并统计，生成 Distance 分布后，同样通过JSD比较生成/真实分布。

---

## 3. **DailyLoc**  
**含义:**  
每日访问过的不同地点（`checkins`）数量，用于度量活跃点的丰富程度。  

**计算方法:**  
- 对于每条轨迹，通过 `len(set(seq['checkins']))` 统计去重后的地点数，计入分布。
- 两组的数据分布通过JSD比较。

---

## 4. **Interval**

**含义:**  
访问不同地点的时间间隔分布。  

**计算方法:**  
- `arrival_times` 为排序后的到达时间序列，将整个序列通过 `np.ediff1d` 求相邻时间点间隔，然后拼接到 `Interval` 统计量。
- 分别统计生成/真实的间隔分布，用JSD作为对比。

---

## 5. **Category**

**含义:**  
按一天 24 个整点时段分辨的类别分布相似性，反映“在不同时段去哪类 POI”。  

**计算方法:**  
- 对每条轨迹，取 `marks`（实际类别，与 `arrival_times` 对齐、时间升序且 ∈[0,24)），按到达时间的整点小时 `h = int(t)` 把每个事件计入该时段的类别计数器：`hourly[h][category] += 1`（`_category_events`）。
- 真实集与生成集各自得到 24 个时段的类别计数分布。
- 对每个时段 `h`，用 `counter_jsd(real_hourly[h], gen_hourly[h])` 计算该时段真实/生成类别分布的 JSD（两边都空 → 0；仅一边空 → `ln 2`；否则在类别并集上用自然对数、`1e-14` 平滑的 `JSD`）。
- `Category = 24 个时段 JSD 的算术平均`（全程无事件则为 NaN）。见 `evaluations/statistical_metrics.py` 的 `temporal_category_metrics`。
- 注：代码中另有一个**全局** `category_jsd`（对整条轨迹 `marks` 的全局类别分布做 JSD）函数，但当前 evaluation v2 的评估入口 `Get_Statistical_Metrics` 报告的 `Category` 用的是上面的**逐时段**版本，并非该全局函数。

---

## **CategoryTransition（类别转移）**

**含义:**  
一阶类别转移分布的相似性，反映“从某一类别出发、下一站去哪一类别”的转移模式。

**计算方法:**  
- 对每条轨迹，按时间顺序取相邻事件的类别对 `(marks[i], marks[i+1])`，按**源类别**累计一阶转移计数：`transitions[left][right] += 1`（`_category_events`）。真实集与生成集各自得到 `real_edges`、`gen_edges`——每个源类别对应一个“下一类别”计数器。
- 源类别全集取两集源类别的并集；对每个源类别 `c`，用 `counter_jsd(real_edges[c], gen_edges[c])` 计算其“下一类别分布”的 JSD（仅一边出现该源类别 → `ln 2`）。
- `CategoryTransition = 各源类别 JSD 的算术平均`（无任何相邻转移则为 NaN）。见 `evaluations/statistical_metrics.py` 的 `temporal_category_metrics`。
- 说明：先按源类别各算一个“转移-目标分布”JSD，再对源类别求平均，而非把所有转移对拼成单一全局分布算一个 JSD。
- 注：在 evaluation v2 的指标名顺序中，`CategoryTransition` 取代了旧表的 `Interval`（`STATISTICAL_NAMES = Distance, Radius, CategoryTransition, DailyLoc, Category, G-RANK`）；上文第 4 节的 `Interval` 为旧口径，当前 v2 报告不再输出。

---

## 6. **G-RANK**  
**含义:**  
最常访问的地点的概率分布的相似性。经常用于衡量热门（Top-K）地标的生成效果。  

**计算方法:**  
- 统计每个地点在所有轨迹被访问的概率（同上，`checkins`）。
- 取按概率排列的前K项（通常K=1000），分别统计生成和真实数据的概率序列，形成概率分布。
- 用 `grank_jsd` 计算两分布的JSD。



## 统计分布相似性变差原因分析以及应对策略

---

### 1. 核心矛盾：语义修正 vs. 空间连贯性

**Distance (移动距离)** 和 **Radius (空间扩散)** 的恶化，揭示了 CDD 投影的一个副作用：**“空间跳变（Teleportation Effect）”**。

*   **Baseline (平滑但错误)**：
    *   模型虽然没学会“A 必须在 B 前面”，但它学好了“人不能瞬间移动”。它生成的轨迹在地理空间上是连贯的、顺滑的（符合人的物理移动规律），因此 Distance/Radius 的分布与真实数据很像（JSD 低）。
*   **CDD (正确但生硬)**：
    *   **机制：** 你的投影算子 $g(\boldsymbol{y})$ 是作用在 **类别 Logits** 上的。它为了满足“存在 A”和“A 在 B 前”，强行修改了某些时间步的类别预测（例如把“公园”改成了“餐馆”）。
    *   **后果：** Marionette 也是生成离散 POI Token 的。当你强行修改类别分布时，模型可能会采样出一个符合该类别、但地理位置极不合理的 POI。
    *   **举例：** 某人原本轨迹是 `家 -> 超市 (1km) -> 健身房 (2km)`。CDD 发现缺了“医院”，强行把中间的“超市”改成了“医院”。但最近的医院可能在 10km 外。
    *   **结果：** 轨迹变成了 `家 -> 医院 (10km) -> 健身房 (10km)`。总距离（Distance）和活动半径（Radius）瞬间爆炸。

**结论：** 这是**“不可避免”**的，只要你的投影算子**只考虑类别逻辑，不考虑地理距离约束**。你是用“空间上的不合理”换取了“逻辑上的合理”。

**避免方法：**

*   **极晚期介入 (Last-k steps)**：正如我们之前讨论的，只在轨迹基本定型时做微调，减少大幅度修改 POI 的机会。
*   **（进阶）加入距离约束：** 在 $g(\boldsymbol{y})$ 中加入一项惩罚，如果修改后的 Token 对应的 GPS 距离上一跳太远，也算违规（这需要可微的 GPS Lookup，实现难度极高，暂不推荐）。
*   **也可以微调原模型**：

---

### 2. 频率失真：局部修正导致的全局偏移

**Category (类别分布)** 和 **G-RANK (热门地点分布)** 的恶化，揭示了 **“概率强行拉升（Artificial Boosting）”** 带来的分布偏移。

*   **真实数据 (Ground Truth)**：A 和 B 的出现是自然发生的，它们嵌入在特定的生活模式中（比如周末才会出现 A+B）。
*   **Baseline**：未能捕捉到这种特定模式，倾向于生成更通用的模式（比如只生成 A 不生成 B）。
*   **CDD**：
    *   **机制：** `existence_weight` 强迫模型在**所有**生成的样本中都要尝试凑齐 A 和 B。
    *   **后果：**
        1.  **频率失真：** 如果测试集中有 100 条轨迹，其中只有 30 条包含 (A, B) 约束。但你在 Batch 采样时，如果给这 100 条都加了强约束，或者约束分配不当，可能会导致 A/B 在生成数据中的**总频率**远高于真实数据。
        2.  **热门点坍缩：** 为了快速满足“存在类别 A”，优化器倾向于把概率推向该类别中**最热门、概率最高**的那个 POI（因为它梯度最大，最容易达成目标）。这会导致生成的 POI 过于集中在头部热门地点，破坏了 G-RANK 的长尾分布特性。

**结论：** 这在一定程度上是**“可避免”**的。

**避免方法：**
*   **软化存在性 (Soft Target)**：不要把 `ReLU(1.0 - count)` 中的目标设为 1.0。
    *   如果设为 **0.8** 或 **0.6**，允许模型以一定的概率生成 A，而不是 100% 必须生成。这能缓解类别频率的过度拉升。
*   **约束分配的准确性**：确保你只对那些“在真实数据中确实包含 A/B”的测试样本施加了 A/B 约束（根据你的代码逻辑，你是从测试集提取约束，这步应该是对的）。
*   **多样性惩罚 (Diversity Penalty)**：防止模型总是生成同一个热门 POI。但这会使优化更复杂。

---

### 3. Interval (时间间隔)：为什么它没怎么变？

*   **观察：** `Interval` 的 JSD 变化很小（0.008 vs 0.007），甚至变好了。
*   **原因：** CDD 投影主要修改的是 **Checkins/Marks (空间/类别)** 序列，而 **Time (时间)** 序列主要由 TPP 模型控制。虽然 Marionette 是联合生成的，但你主要在修改 POI 的 Logits，对时间维度的干扰较小。这说明你的投影方法**精确地靶向**了它该管的地方，没有由于梯度回传错误地破坏时间结构。这是好消息。

---

### 总结建议

既然目标是**复原真实数据的分布**（即 100% 满足约束，且 Distance/Category JSD 很低），目前的策略中：

1.  **Distance/Radius 变差** 是因为**“为了逻辑牺牲了地理连贯性”**。
    *   **对策：** 坚定执行 **`projection_last_k_steps=30`** (或更小) + **`projection_frequency=1`** 的策略。只在最后时刻做微小的“替换”，而不是重构轨迹。

2.  **Category/G-RANK 变差** 是因为**“为了存在性牺牲了多样性”**。
    *   **对策：** 尝试降低 `projection_existence_weight` 的 **Target**。例如将 `F.relu(1.0 - count)` 改为 `F.relu(0.8 - count)`。这告诉模型：“只要你有 80% 的意愿生成 A 就可以了，不需要强行拉满到 100%”，这样能保留更多原始分布的特征。



## Baselines

### Baseline1

Marionette：

### Baseline 2: Post-hoc Swap (后处理交换)

**核心思想**：
先让模型自由生成 POI 序列，完全忽略约束。生成结束后，通过一个确定性的后处理算法，强制交换违反约束的 POI 位置以满足偏序关系。

**方法过程**：
1.  **自由生成**：使用训练好的扩散模型（`DiscreteDiffusion`）进行标准的无条件采样，生成原始 POI 序列。此过程不施加任何约束引导。
2.  **约束检测**：对于每个生成的序列，解析其对应的偏序约束矩阵（PO Matrix）。
3.  **拓扑排序与交换**：
    *   检查生成的序列是否满足偏序关系（例如：POI A 必须在 POI B 之前）。
    *   如果发现违规（例如 B 在 A 之前），则直接在序列中交换这两个点的位置，或者根据拓扑排序的结果重新排列涉及冲突的子序列。
4.  **最终输出**：输出经过“物理修正”后的序列。这种方法能保证约束满足率极高（通常 100%），但可能会破坏序列的时间连续性和地理合理性（因为是硬性交换）。

---

### Baseline 3: Classifier-Based Guidance (基于分类器的引导)

**核心思想**：
在扩散生成的每一步去噪过程中，利用一个“能量函数”（Energy Function）来评估当前中间状态违背约束的程度，并计算该能量对中间状态的梯度。利用这个梯度来修改去噪方向，引导生成过程向满足约束的方向偏移。

**方法过程**：
1.  **能量定义**：定义一个损失函数（Energy），衡量当前生成的 Logits 与目标约束的偏差。
    *   *存在性违规*：目标 POI 的预测概率之和是否达到阈值。
    *   *顺序违规*：违反偏序关系的 POI 对的累积概率质量。
2.  **去噪循环**：在扩散模型的每一步去噪（`p_sample`）中：
    *   计算当前预测的 $p(x_0 | x_t)$。
    *   计算该预测的能量值 $E(x_0)$。
    *   对 Logits 求关于能量的梯度 $\nabla_{\text{logits}} E$。
3.  **梯度引导**：使用公式 $\text{Logits}_{\text{new}} = \text{Logits}_{\text{old}} - \text{scale} \times \nabla E$ 更新预测结果。
4.  **参数控制**：通过 `guidance_scale` 控制引导力度，通过 `temperature` 控制梯度的尖锐程度。不需要重新训练模型，但计算开销大（每一步都要反向传播）。

---

### Baseline 4: Classifier-Free Guidance (CFG, 无分类器引导)

**核心思想**：
重新训练扩散模型，使其能够同时学习“有约束生成”和“无约束生成”的分布。在采样时，通过线性组合“有条件预测”和“无条件预测”的 Logits，人为放大约束条件的影响力，从而生成符合约束的序列。

**方法过程**：
1.  **模型重训 (Retraining)**：
    *   将偏序矩阵（PO Matrix）作为条件编码（Embedding）输入模型。
    *   **Condition Dropout**：在训练过程中，以一定概率将 PO Matrix 条件替换为 Null Embedding（全零或可学习向量），强迫模型学习无条件的先验分布。
2.  **混合采样**：
    *   在生成的每一步，模型进行两次前向传播（Forward Pass）：
        *   **Conditional**：输入真实的 PO Matrix，得到 $\text{Logits}_{\text{cond}}$。
        *   **Unconditional**：输入 Null Embedding，得到 $\text{Logits}_{\text{uncond}}$。
3.  **Logits 组合**：
    *   应用 CFG 公式：$\text{Logits}_{\text{final}} = \text{Logits}_{\text{uncond}} + \text{scale} \times (\text{Logits}_{\text{cond}} - \text{Logits}_{\text{uncond}})$。
    *   当 `scale > 1` 时，模型会被推向更加符合约束的方向，同时远离无条件的通用生成结果。
4.  **优势**：无需在采样时计算梯度，推理速度快于 Baseline 3，且通常生成的质量更稳定，不易出现 Baseline 3 中的对抗性噪声破坏结构的问题。

以下是所有方法的实验结果汇总表格。

### 实验结果对比汇总 (Experiment Summary)

​                                                                                                                                                                                                                                          

##### baseline1                                                                                                                                                                 

Distance: 0.06158823103653222, Radius: 0.021997760260718664, Interval: 0.008171208551622698, DailyLoc: 0.035915595729734656, Category: 0.002068002310945113, 'G-RANK': 0.11384010039526779, ==OVR_ref_skip: 0.46834525941498606, OVR_ref_strict: 0.9051507287705348, coverage: 0.16588226217112878, Unsat_ref: 0.9434457217807268== 

##### baseline2  

Distance: 0.07247314602247641, Radius: 0.02206092560193211, Interval: 0.03469258037329195, DailyLoc: 0.03431742599182735, Category: 0.0026133273590957934, 'G-RANK': 0.11662059109808587, ==OVR_ref_skip: 0.0, OVR_ref_strict: 0.8258163969007343, coverage: 0.15654656276521872, Unsat_ref: 0.8434534372347813== 

##### baseline3 scale=5000000 projection_existence_weight：10.0 gumbel_temperature:1.0 

Distance: 0.11141896718931404, Radius: 0.056178027383742876, Interval: 0.007387125784703965, DailyLoc: 0.03598737108668015, Category: 0.009371708634891958, 'G-RANK': 0.15051662758949266,                                                     ==OVR_ref_skip: 0.44301493582226315, OVR_ref_strict: 0.757085051390413, coverage: 0.39950621094051386, Unsat_ref: 0.8194583751253761==           

##### baseline4 cond_dropout_rate: 0.0 Scale:1.0

Distance: 0.18355314887335064, Radius: 0.09331312525074992, Interval: 0.007199457352294503, DailyLoc: 0.040061247191114466, Category: 0.0007550952962749516, 'G-RANK': 0.12784483528160295,                                          ==OVR_ref_skip: 0.1440859340529535, OVR_ref_strict: 0.2827518359798709, coverage: 0.7885965589074917, Unsat_ref: 0.4005092199675951==



##### projection_outer_iters:20

Distance: 0.13765354396143836, Radius: 0.07748723553621606, Interval: 0.008221829020761974, DailyLoc: 0.03633080029537968, Category: 0.019336851907236923, 'G-RANK': 0.18198385047845422,                                           ==OVR_ref_skip: 0.14362201894406462, OVR_ref_strict: 0.37025626254172017, coverage: 0.6624488851168892, Unsat_ref: 0.4419412082401049==

##### #CDD projection_last_k_steps:100 projection_inner_iters: 100    projection_mu_alpha: 2.0        # α，放大 μ #projection_outer_iters:100 #projection_frequency: 4 #projection_eta: 0.4             # η

Distance: 0.17154447590288185, Radius: 0.10623637490620774, Interval: 0.007554740057441065, DailyLoc: 0.0416628370330472, Category: 0.04284067181929395, 'G-RANK': 0.22968598147630903, ==OVR_ref_skip: 0.4115939510510596, OVR_ref_strict: 0.5521678551120002, coverage: 0.7332767533369339, Unsat_ref: 0.5798935267340483==

##### CDD with independent multipliers

Distance: 0.16367650705076497, Radius: 0.10082774261376902, Interval: 0.00757942255986814, DailyLoc: 0.03668562142969747, Category: 0.019703085174473542, 'G-RANK': 0.1921009583651329, ==OVR_ref_skip: 0.2120599622142832, OVR_ref_strict: 0.45256091593102793, coverage: 0.6171591698171437, Unsat_ref: 0.5352210477586606==

##### projection_eta：0.15

##### projection_inner_iters:25

##### projection_outer_iters:25

Distance: 0.14360421045257377, Radius: 0.08255073071623206, Interval: 0.007861834907668935, DailyLoc: 0.03487371826946921, Category: 0.019958239973672923, 'G-RANK': 0.18489533048699416,                                           ==OVR_ref_skip: 0.13865519113245672, OVR_ref_strict: 0.3718606839988478, coverage: 0.6514929403595402, Unsat_ref: 0.4509682894838361==

### 重新划分数据集重新训练

##### CDD

Distance: 0.14204938375565335, Radius: 0.07507984669891674, Interval: 0.004499645011982916, DailyLoc: 0.03605988746187773, Category: 0.005589959171283337, 'G-RANK': 0.1885707815529386, OVR_ref_skip: 0.19768760112526831, OVR_ref_strict: 0.5191231375181946, coverage: 0.5457285330703052, Unsat_ref: 0.5748808153871445

##### projection_gumbel_temperature：3.0
Distance: 0.17712643842276254, Radius: 0.1049418162916293, Interval: 0.004738307586134519, DailyLoc: 0.042884739054163915, Category: 0.016509137399321257, 'G-RANK': 0.23697031342674052, OVR_ref_skip: 0.1649948675062311, OVR_ref_strict: 0.4461852796050188, coverage: 0.6169105156446929, Unsat_ref: 0.5051235684147076

##### baseline4

Distance: 0.15926182075259454, Radius: 0.09155329472363077, Interval: 0.004741506914698266, DailyLoc: 0.03974883947170077, Category: 0.003338713762571789, 'G-RANK': 0.17283506510441676, OVR_ref_skip: 0.38368167887107274, OVR_ref_strict: 0.5927781062278034, coverage: 0.6129102964546003, Unsat_ref: 0.6841470765521398

##### baseline2 OOD data retrain
Distance: 0.06891240163875399, Radius: 0.020396111548844166, Interval: 0.038545979490379925, DailyLoc: 0.047744029736314725, Category: 0.03239618715471512, 'G-RANK': 0.19564706689223466, OVR_ref_skip: 0.0, OVR_ref_strict: 0.8612546517514817, coverage: 0.14954244068168118, Unsat_ref: 0.8504575593183188

##### baseline1 OOD data retrain
Distance: 0.061021493945879315, Radius: 0.020396111548844166, Interval: 0.004766970284762686, DailyLoc: 0.047744029736314725, Category: 0.03239618715471512, 'G-RANK': 0.19564706689223466, OVR_ref_skip: 0.4817600444152118, OVR_ref_strict: 0.9296572397984179, coverage: 0.14954244068168118, Unsat_ref: 0.9549016384459422

##### baseline3 OOD data retrain

Distance: 0.15414438663585342, Radius: 0.08652264118745508, Interval: 0.004781397848402362, DailyLoc: 0.03879522683644168, Category: 0.008722727080775133, 'G-RANK': 0.20335224258011173, OVR_ref_skip: 0.2809034594075819, OVR_ref_strict: 0.8021929828367249, coverage: 0.2643432516850238, Unsat_ref: 0.8263466491314593

### new distance and new coverage

#### CDD

Distance: 0.14842450911185345, Radius: 0.09886548255335678, Interval: 0.004710430834719498, DailyLoc: 0.04124924997253542, Category: 0.017815683406667726, 'G-RANK': 0.23892487051889083, OVR_ref_skip: 0.15876391323190184, OVR_ref_strict: 0.44012047136006727, coverage: 0.8007179026985449, Unsat_ref: 0.5014521343635268

#### baseline4

Distance: 0.19560651503698354, Radius: 0.0927267349938959, Interval: 0.004380694385776995, DailyLoc: 0.03293341082165363, Category: 0.003046145039075917, 'G-RANK': 0.17620330401114165, OVR_ref_skip: 0.3800539452975051, OVR_ref_strict: 0.5951124001713175, coverage: 0.7910390359592334, Unsat_ref: 0.6850238369225711

#### baseline1

Distance: 0.1306504839290164, Radius: 0.020396111548844166, Interval: 0.004766970284762686, DailyLoc: 0.047744029736314725, Category: 0.03239618715471512, 'G-RANK': 0.19564706689223466, OVR_ref_skip: 0.4817600444152118, OVR_ref_strict: 0.9296572397984179, coverage: 0.4306775206717518, Unsat_ref: 0.9549016384459422

#### baseline2

Distance: 0.14543747662981776, Radius: 0.020396111548844166, Interval: 0.038545979490379925, DailyLoc: 0.047744029736314725, Category: 0.03239618715471512, 'G-RANK': 0.19564706689223466, OVR_ref_skip: 0.0, OVR_ref_strict: 0.8612546517514817, coverage: 0.4306775206717518, Unsat_ref: 0.8504575593183188

#### baseline3

Distance: 0.13794190327282985, Radius: 0.05599765743300754, Interval: 0.004768468937834069, DailyLoc: 0.03471773924659513, Category: 0.010288504457849913, 'G-RANK': 0.18804944079920988, OVR_ref_skip: 0.24485029777768547, OVR_ref_strict: 0.8278687786915899, coverage: 0.5336837382219088, Unsat_ref: 0.8273878020713463



---

### 选取新的数据集进行实验

#### 新数据集 (New York City) 的引入与深度预处理 (Data Preprocessing)

##### 数据集简介

This dataset includes long-term (about 10 months) check-in data in New York city and Tokyo collected from Foursquare from 12 April 2012 to 16 February 2013.
It contains a file in tsv format with 8 columns, which are:

1. User ID (anonymized)
2. Venue ID (Foursquare)
3. Venue category ID (Foursquare)
4. Venue category name (Fousquare)
5. Latitude
6. Longitude
7. Timezone offset in minutes (The offset in minutes between when this check-in occurred and the same time in UTC)
8. UTC time

The file dataset_TSMC2014_NYC.txt contains 227428 check-ins in New York city.

数据预处理：
1. **长轨迹切分 (Trajectory Segmentation)**：原始 NYC 数据是长达 10 个月的连续记录。通过引入 **24小时时间窗口（Time Gap）** 阈值，将庞大的单用户历史成功切分为几万条符合人类单次/短期出行模式的合理子轨迹。
2. **NYC 本地化 Context 构造**：针对纽约当地情况，使用 Pandas 重构了环境条件（Conditions）。
3. **细粒度类别语义降维 (Semantic Category Mapping)**：NYC 原始数据包含 400 个细粒度 POI 类别，这会导致 $400 \times 400$ 的偏序矩阵维度爆炸且极度稀疏。通过编写语义映射规则，将其**聚合为了 9 个一级大类**，与 Istanbul 格式对齐。
4. **长尾 POI 过滤 (Low-frequency POI Filtering)**：NYC 原始 POI 数量高达 3.8 万，导致扩散模型在全词表 Softmax 时显存溢出。通过设置签到频次阈值（$\ge 10$次），剔除了低频噪点，将 POI 词表缩减至 4000 左右，大幅降低了计算开销并提升了数据质量。

#### 三、 偏序约束矩阵生成与 OOD 切分 (Constraint Matrix & OOD Split)
基于清洗好的纽约数据，完成了模型所需的特征工程：
1. **生成全局特征**：编写并运行了 NYC 专属的 PO Matrix 脚本，成功提取了 9×9 的偏序邻接矩阵。
2. **零样本评估集划分 (Zero-shot OOD Split)**：利用 `split_ood_dataset.py`，成功剥离了部分未见过的偏序约束（Unseen Constraints）专用于测试集，构建了 `NewYork_PO1_OOD` 数据集，确保训练集与测试集在偏序关系上存在巨大差异（Domain Gap），以验证模型的泛化/零样本推断能力。
3. 划分后数据集概览

<img src="C:/Users/19542/AppData/Roaming/Typora/typora-user-images/image-20260506153045472.png" alt="image-20260506153045472" style="zoom:80%;" />

​	![image-20260506153755688](C:/Users/19542/AppData/Roaming/Typora/typora-user-images/image-20260506153755688.png)

---

###  下一步计划 (Next Steps)

使用此数据集将所有baseline跑一遍

# 实验结果总汇

整合日期：2026-10-05。范围：当前工作区与前序计划中的五组正式实验/工程校准。

**结论：双城市主实验已完成 60/60；计入早期单种子 OOD 对照后，去重共有 65 份 OOD 生成结果。另有 10 组训练集工程校准，不计入 OOD 成绩。**

## 从这里阅读

- [7页综合报告与表格 PDF](<experiment-summary.pdf>)
- [双城市最终完整报告](<sources/two-city-v1/report.md>)
- [逐结果索引（含本地路径、哈希、种子和分片）](<result-index.json>)
- [可筛选长表 CSV（原始最终文件，未改写）](<sources/two-city-v1/summary-long.csv>)
- [LaTeX 四表预览源文件](<sources/two-city-v1/table-preview.tex>)
- [本地核验](<local-audit.json>) / [服务器核验](<server-audit.json>) / [文件清单](<file-inventory.json>)

## 实验关系与去重

| 实验组                  | 有效结果 | 本次汇总角色                                   | 来源                                                         |
| ----------------------- | -------: | ---------------------------------------------- | ------------------------------------------------------------ |
| 两城市统一对比与消融    |       60 | 主结果；两城市各 30，39 新增 + 21 复用         | [最终报告](<sources/two-city-v1/report.md>)                  |
| NewYork 三种子消融      |       21 | 全部已进入上述 60 份，不再累加                 | [消融报告](<../pcdg-ablation-v1/report.md>)                  |
| NewYork 单种子基线      |   3 新增 | 历史参照；表内 native/projection 复用早期 2 份 | [基线报告](<../nyood-baselines-v1-20261002/report.md>)       |
| NewYork 完整 OOD 重采样 |        2 | 历史 native/projection 对照                    | [重采样报告](<../nyood1000s13539820261001_perfcal-v1-ood-r2/report.md>) |
| 训练集工程校准          |       10 | 128/512 条训练条件；不能作为 OOD 成绩          | [校准报告](<../../calibration_runs/perfcal-v1/report.md>)    |

JointGen = No Projection，PCDG = Full：每个城市的对应三份文件及哈希完全相同，只是表中别名。两城市主表共有 20 个独立“城市 × 方法/变体”组合，每个组合 3 个采样种子。

## 主要结果

| 数据集           | JointGen 严格违反率 | PCDG 严格违反率 | 降低（百分点） | PCDG 类别对覆盖率 |
| ---------------- | ------------------: | --------------: | -------------: | ----------------: |
| Istanbul_PO1_OOD |       92.89 ± 0.08% |   32.80 ± 0.50% |          60.09 |     70.99 ± 0.08% |
| NewYork_PO1_OOD  |       89.45 ± 0.20% |   38.51 ± 0.95% |          50.94 |     67.60 ± 1.86% |

- 五方法对比中，PCDG 在两城市的严格违反率、类别对覆盖率、类别覆盖率及 Unsat 均值最好，但并非所有分布相似性指标最好。
- PO-CFG 在 NewYork 的 6 个分布指标中有 5 个均值最优；Istanbul 的 Distance、G-RANK 最优。约束满足与分布拟合存在取舍。
- PostSwap 的 OVR_skip 为 0，但不能补回缺失类别；两城市严格违反率仍较高，不能把 skip=0 解读为全部约束满足。
- 消融结果支持存在性项、顺序项和乘子更新在当前协议中的作用；去 KL、去 Gumbel 并未一致恶化结果，不应宣称所有模块均有显著收益。
- 以上是均值层面的描述，未做显著性检验。三种子样本标准差只反映固定检查点的采样随机性。

## 表中方法简要说明

以下说明对应 2026-10-05 汇总的双城市实验，适用于后面的分布相似性表和约束满足表；前文历史实验的参数与实现不一定完全相同。

### 方法对比表中的 5 个方法

| 方法 | 简要说明 |
| --- | --- |
| JointGen | 原始时空生成基线：保留 Marionette 的上下文、时间与空间生成流程，不加入偏序条件编码、约束引导或投影，用于衡量无偏序控制时的生成表现。 |
| PostSwap | 生成后重排：先用 JointGen 生成轨迹，再按偏序关系重排已出现的类别及其 POI；不增加缺失类别，因此修正已有类别的顺序不等于满足全部约束。 |
| EnergyGuide | 能量梯度引导：在选定的去噪步，以“顺序违规 + 5 × 存在性违规”为能量，对预测 logits 做一次梯度修正；不执行 ALM 多轮投影，也不额外训练分类器。 |
| PO-CFG (s=1) | 偏序条件生成：将偏序矩阵编码为模型条件，单独训练空间模型，并通过随机丢弃偏序条件学习有条件与空条件两种预测；采样时 s=1，正常使用偏序条件，但不额外放大其影响。 |
| PCDG | 完整约束投影方法：不重新训练基座模型，在采样阶段通过增广拉格朗日法（ALM）反复修正预测 logits，同时考虑类别存在性与顺序约束，并用 KL 项限制对原预测分布的偏离。 |

其中，PO-CFG 的 $s$ 是引导强度，当前实现采用：

$$
\ell_s=\ell_{\varnothing}+s(\ell_{\mathcal{S}}-\ell_{\varnothing})
$$

$\ell_{\mathcal{S}}$ 表示有偏序条件的 logits，$\ell_{\varnothing}$ 表示去掉偏序条件后的 logits；两者均保留其他上下文与时间条件。$s=1$ 时输出就是 $\ell_{\mathcal{S}}$，不是“不使用条件”，也不是采样次数；$s>1$ 才额外增强条件影响，但不保证指标更好。

### 消融表中的 7 个变体

消融以 Full 为参照，在固定基座检查点和采样协议下关闭相应组件，不重新训练模型。

| 变体 | 简要说明 |
| --- | --- |
| No Projection | 完全关闭约束投影，直接使用基座模型采样；与方法对比表中的 JointGen 是同一组结果。 |
| No Existence | 去掉存在性惩罚，保留顺序项与 KL 项；用于检验显式要求约束类别出现的作用，避免仅因类别缺失而看似没有错序。 |
| No Order | 去掉顺序惩罚，保留存在性项与 KL 项；仍促使所需类别出现，但不显式约束它们的前后顺序。 |
| No KL | 去掉保持原预测分布的 KL 项，保留存在性、顺序约束及 ALM 更新；用于检验分布保真项的影响。 |
| Fixed Multipliers | 保留投影与各损失项，但固定拉格朗日乘子 $\lambda$ 和罚参数 $\mu$，不随违规情况更新；用于检验自适应约束惩罚的作用。 |
| No Gumbel Noise | 投影计算中去掉 Gumbel 随机噪声，仍使用带温度的 Softmax 松弛；仅关闭投影噪声，不关闭扩散采样本身的随机性。 |
| Full | 完整 PCDG：保留存在性项、顺序项、KL 项、乘子与罚参数更新，以及 Gumbel-Softmax 松弛；与方法对比表中的 PCDG 是同一组结果。 |

Full 的固定设置为：顺序权重 1、存在性权重 5、KL 权重 1、投影温度 3，每个选定去噪步使用 10 轮外循环 × 50 轮内优化，不提前停止。以上说明描述组件用途，不意味着每个组件都必然改善所有指标。

## 双城市主表与消融表

### 方法对比：分布相似性

### Istanbul_PO1_OOD

| Method | Distance ↓ | Radius ↓ | CategoryTransition ↓ | DailyLoc ↓ | Category ↓ | G-RANK ↓ |
|---|---:|---:|---:|---:|---:|---:|
| JointGen | 0.1128 ± 0.0087 | 0.0226 ± 0.0017 | 0.0403 ± 0.0001 | 0.0414 ± 0.0009 | 0.0649 ± 0.0003 | 0.1922 ± 0.0022 |
| PostSwap | 0.0949 ± 0.0127 | 0.0226 ± 0.0017 | 0.0361 ± 0.0005 | 0.0414 ± 0.0009 | 0.0647 ± 0.0018 | 0.1922 ± 0.0022 |
| EnergyGuide | 0.1156 ± 0.0052 | **0.0222 ± 0.0025** | 0.0424 ± 0.0009 | 0.0392 ± 0.0020 | 0.0643 ± 0.0016 | 0.1921 ± 0.0004 |
| PO-CFG (s=1) | **0.0890 ± 0.0050** | 0.0263 ± 0.0020 | 0.0365 ± 0.0008 | 0.0444 ± 0.0022 | 0.0630 ± 0.0007 | **0.1794 ± 0.0026** |
| PCDG | 0.1314 ± 0.0035 | 0.0516 ± 0.0022 | **0.0162 ± 0.0006** | 0.0368 ± 0.0014 | 0.0446 ± 0.0020 | 0.1820 ± 0.0016 |
| 方案A（类别一致解码） | 0.1539 ± 0.0028 | 0.0892 ± 0.0024 | 0.0238 ± 0.0011 | **0.0297 ± 0.0012** | **0.0423 ± 0.0009** | 0.1813 ± 0.0018 |

### NewYork_PO1_OOD

| Method | Distance ↓ | Radius ↓ | CategoryTransition ↓ | DailyLoc ↓ | Category ↓ | G-RANK ↓ |
|---|---:|---:|---:|---:|---:|---:|
| JointGen | 0.2904 ± 0.0156 | 0.1104 ± 0.0071 | 0.0070 ± 0.0008 | 0.0409 ± 0.0035 | 0.0184 ± 0.0002 | 0.1612 ± 0.0013 |
| PostSwap | 0.2276 ± 0.0448 | 0.1104 ± 0.0071 | 0.0070 ± 0.0007 | 0.0409 ± 0.0035 | 0.0168 ± 0.0006 | 0.1612 ± 0.0013 |
| EnergyGuide | 0.2943 ± 0.0060 | 0.1112 ± 0.0061 | 0.0071 ± 0.0011 | 0.0411 ± 0.0029 | 0.0183 ± 0.0003 | 0.1622 ± 0.0009 |
| PO-CFG (s=1) | **0.1228 ± 0.0160** | **0.0823 ± 0.0029** | **0.0053 ± 0.0018** | 0.0499 ± 0.0008 | **0.0121 ± 0.0007** | **0.1525 ± 0.0025** |
| PCDG | 0.2210 ± 0.0224 | 0.1265 ± 0.0051 | 0.0132 ± 0.0004 | 0.0351 ± 0.0035 | 0.0173 ± 0.0013 | 0.1942 ± 0.0041 |
| 方案A（类别一致解码） | 0.1989 ± 0.0198 | 0.1681 ± 0.0042 | 0.0182 ± 0.0012 | **0.0335 ± 0.0022** | 0.0178 ± 0.0012 | 0.1914 ± 0.0059 |

> 「方案A（类别一致解码）」为 PCDG 之上追加类别一致 POI 解码的测试侧三种子结果（ny-ist-category-formal-v1c-20261011，off 复用封存 Full≈PCDG 行，v2 口径一致、可比）；详见文末专节。

[对应 LaTeX 表片段](<sources/two-city-v1/methods-jsd.tex>)

### 方法对比：约束满足

### Istanbul_PO1_OOD

| Method | OVR_strict ↓ | OVR_skip ↓ | Pair coverage ↑ | Category coverage ↑ | Unsat ↓ |
|---|---:|---:|---:|---:|---:|
| JointGen | 92.89 ± 0.08 | 48.05 ± 0.74 | 14.63 ± 0.19 | 43.19 ± 0.20 | 95.57 ± 0.15 |
| PostSwap | 86.10 ± 0.15 | **0.00 ± 0.00** | 14.63 ± 0.19 | 43.19 ± 0.20 | 85.37 ± 0.19 |
| EnergyGuide | 90.66 ± 0.06 | 44.76 ± 0.58 | 19.17 ± 0.43 | 47.06 ± 0.25 | 92.89 ± 0.16 |
| PO-CFG (s=1) | 63.45 ± 0.23 | 19.09 ± 0.54 | 43.72 ± 0.35 | 66.00 ± 0.23 | 69.07 ± 0.45 |
| PCDG | 32.80 ± 0.50 | 9.12 ± 0.10 | 70.99 ± 0.08 | 85.36 ± 0.15 | 36.86 ± 0.37 |
| 方案A（类别一致解码） | **3.72 ± 0.13** | 1.76 ± 0.10 | **94.74 ± 0.16** | **97.81 ± 0.07** | **6.90 ± 0.18** |

### NewYork_PO1_OOD

| Method | OVR_strict ↓ | OVR_skip ↓ | Pair coverage ↑ | Category coverage ↑ | Unsat ↓ |
|---|---:|---:|---:|---:|---:|
| JointGen | 89.45 ± 0.20 | 52.92 ± 0.86 | 21.13 ± 0.15 | 46.75 ± 0.12 | 93.64 ± 0.26 |
| PostSwap | 78.22 ± 0.48 | **0.00 ± 0.00** | 21.13 ± 0.15 | 46.75 ± 0.12 | 78.87 ± 0.15 |
| EnergyGuide | 89.17 ± 0.14 | 51.59 ± 0.88 | 21.17 ± 0.19 | 47.14 ± 0.14 | 93.20 ± 0.39 |
| PO-CFG (s=1) | 41.91 ± 1.37 | 8.97 ± 0.88 | 59.92 ± 1.58 | 74.99 ± 1.08 | 49.48 ± 1.20 |
| PCDG | 38.51 ± 0.95 | 14.11 ± 1.22 | 67.60 ± 1.86 | 81.54 ± 0.70 | 42.42 ± 1.84 |
| 方案A（类别一致解码） | **9.52 ± 1.04** | 3.47 ± 0.64 | **88.67 ± 1.74** | **93.92 ± 0.68** | **14.09 ± 2.11** |

> 「方案A（类别一致解码）」为在 PCDG 之上追加类别一致 POI 解码的测试侧三种子结果（运行 ny-ist-category-formal-v1c-20261011，off 复用封存 Full≈上面 PCDG 行；约束指标口径一致、可比）。详见文末「方案A：类别一致解码 off/on 对照」小节。

[对应 LaTeX 表片段](<sources/two-city-v1/methods-constraints.tex>)

### 消融实验：分布相似性

### Istanbul_PO1_OOD

| Method | Distance ↓ | Radius ↓ | CategoryTransition ↓ | DailyLoc ↓ | Category ↓ | G-RANK ↓ |
|---|---:|---:|---:|---:|---:|---:|
| No Projection | 0.1128 ± 0.0087 | 0.0226 ± 0.0017 | 0.0403 ± 0.0001 | 0.0414 ± 0.0009 | 0.0649 ± 0.0003 | 0.1922 ± 0.0022 |
| No Existence | **0.1084 ± 0.0114** | **0.0215 ± 0.0027** | 0.0389 ± 0.0005 | 0.0412 ± 0.0021 | 0.0661 ± 0.0009 | 0.1924 ± 0.0028 |
| No Order | 0.1176 ± 0.0023 | 0.0486 ± 0.0035 | 0.0227 ± 0.0008 | **0.0358 ± 0.0006** | 0.0439 ± 0.0018 | **0.1770 ± 0.0040** |
| No KL | 0.1406 ± 0.0117 | 0.0574 ± 0.0028 | 0.0164 ± 0.0006 | 0.0361 ± 0.0012 | **0.0429 ± 0.0012** | 0.1817 ± 0.0024 |
| Fixed Multipliers | 0.1296 ± 0.0093 | 0.0526 ± 0.0045 | 0.0185 ± 0.0005 | 0.0361 ± 0.0009 | 0.0457 ± 0.0025 | 0.1841 ± 0.0020 |
| No Gumbel Noise | 0.1315 ± 0.0091 | 0.0524 ± 0.0023 | **0.0160 ± 0.0010** | 0.0370 ± 0.0005 | 0.0439 ± 0.0023 | 0.1831 ± 0.0027 |
| Full | 0.1314 ± 0.0035 | 0.0516 ± 0.0022 | 0.0162 ± 0.0006 | 0.0368 ± 0.0014 | 0.0446 ± 0.0020 | 0.1820 ± 0.0016 |

### NewYork_PO1_OOD

| Method | Distance ↓ | Radius ↓ | CategoryTransition ↓ | DailyLoc ↓ | Category ↓ | G-RANK ↓ |
|---|---:|---:|---:|---:|---:|---:|
| No Projection | 0.2904 ± 0.0156 | 0.1104 ± 0.0071 | **0.0070 ± 0.0008** | 0.0409 ± 0.0035 | 0.0184 ± 0.0002 | 0.1612 ± 0.0013 |
| No Existence | 0.2439 ± 0.0277 | **0.1004 ± 0.0053** | 0.0075 ± 0.0008 | 0.0393 ± 0.0044 | 0.0189 ± 0.0004 | **0.1587 ± 0.0006** |
| No Order | 0.2175 ± 0.0284 | 0.1196 ± 0.0108 | 0.0138 ± 0.0003 | 0.0352 ± 0.0029 | 0.0177 ± 0.0003 | 0.1860 ± 0.0048 |
| No KL | 0.2415 ± 0.0239 | 0.1421 ± 0.0046 | 0.0125 ± 0.0006 | 0.0352 ± 0.0041 | 0.0173 ± 0.0013 | 0.1939 ± 0.0064 |
| Fixed Multipliers | 0.2402 ± 0.0487 | 0.1218 ± 0.0018 | 0.0127 ± 0.0005 | 0.0353 ± 0.0041 | 0.0175 ± 0.0006 | 0.1891 ± 0.0023 |
| No Gumbel Noise | **0.2055 ± 0.0205** | 0.1248 ± 0.0056 | 0.0130 ± 0.0010 | 0.0360 ± 0.0036 | 0.0173 ± 0.0003 | 0.1918 ± 0.0041 |
| Full | 0.2210 ± 0.0224 | 0.1265 ± 0.0051 | 0.0132 ± 0.0004 | **0.0351 ± 0.0035** | **0.0173 ± 0.0013** | 0.1942 ± 0.0041 |

[对应 LaTeX 表片段](<sources/two-city-v1/ablation-jsd.tex>)

### 消融实验：约束满足

### Istanbul_PO1_OOD

| Method | OVR_strict ↓ | OVR_skip ↓ | Pair coverage ↑ | Category coverage ↑ | Unsat ↓ |
|---|---:|---:|---:|---:|---:|
| No Projection | 92.89 ± 0.08 | 48.05 ± 0.74 | 14.63 ± 0.19 | 43.19 ± 0.20 | 95.57 ± 0.15 |
| No Existence | 93.67 ± 0.15 | **7.67 ± 0.56** | 7.34 ± 0.12 | 36.92 ± 0.19 | 93.58 ± 0.17 |
| No Order | 63.50 ± 0.76 | 50.34 ± 0.73 | **72.53 ± 0.70** | **86.08 ± 0.17** | 70.70 ± 0.73 |
| No KL | 32.85 ± 0.52 | 9.20 ± 0.29 | 71.11 ± 0.08 | 85.40 ± 0.16 | 36.75 ± 0.57 |
| Fixed Multipliers | 42.70 ± 0.51 | 20.97 ± 0.90 | 71.25 ± 0.89 | 85.00 ± 0.42 | 45.64 ± 0.58 |
| No Gumbel Noise | **32.75 ± 0.39** | 9.13 ± 0.22 | 70.97 ± 0.34 | 85.38 ± 0.19 | **36.75 ± 0.25** |
| Full | 32.80 ± 0.50 | 9.12 ± 0.10 | 70.99 ± 0.08 | 85.36 ± 0.15 | 36.86 ± 0.37 |

### NewYork_PO1_OOD

| Method | OVR_strict ↓ | OVR_skip ↓ | Pair coverage ↑ | Category coverage ↑ | Unsat ↓ |
|---|---:|---:|---:|---:|---:|
| No Projection | 89.45 ± 0.20 | 52.92 ± 0.86 | 21.13 ± 0.15 | 46.75 ± 0.12 | 93.64 ± 0.26 |
| No Existence | 90.38 ± 0.47 | 18.04 ± 1.26 | 10.95 ± 0.45 | 39.47 ± 0.12 | 91.80 ± 0.40 |
| No Order | 65.43 ± 0.12 | 51.24 ± 0.30 | **69.92 ± 0.74** | **82.84 ± 0.40** | 72.86 ± 0.94 |
| No KL | **38.24 ± 0.96** | 13.98 ± 0.82 | 67.67 ± 1.56 | 81.57 ± 0.72 | **42.40 ± 1.48** |
| Fixed Multipliers | 58.45 ± 0.66 | 35.14 ± 0.60 | 63.79 ± 0.76 | 78.37 ± 0.35 | 60.67 ± 0.49 |
| No Gumbel Noise | 39.11 ± 1.53 | **13.93 ± 1.34** | 66.41 ± 1.32 | 80.95 ± 0.65 | 43.91 ± 1.59 |
| Full | 38.51 ± 0.95 | 14.11 ± 1.22 | 67.60 ± 1.86 | 81.54 ± 0.70 | 42.42 ± 1.84 |

[对应 LaTeX 表片段](<sources/two-city-v1/ablation-constraints.tex>)

这里不是“最小值到最大值的范围”，而是 **平均值 ± 标准差**。以 `52.92 ± 0.86` 为例：- **52.92%**：三次采样结果的平均值。- **0.86 个百分点**：三次结果的样本标准差，描述波动大小；越小，说明这三次结果越接近。之所以统一这样展示，是因为原实验方案要求：**固定训练好的模型，用 3 个不同随机种子分别生成轨迹，再汇总指标**。不是重新训练了三次模型。所以，主要成绩就是 **52.92%**，后面的数字说明采样波动。它不代表结果一定落在 `52.06%～53.78%` 内，也不是置信区间。

---

## 方案A：类别一致解码 off/on 对照（测试侧三种子，2026-10-12）

**运行** `ny-ist-category-formal-v1c-20261011`（双城、双卡、代码提交 `4462f90`）。off = 复用封存 Full（two-city-distance-v2，未重采）≈ 上表 PCDG；on = 在其基础上开启"类别一致 POI 解码"（sampled-category-poi-v2）。

**口径**
- 评估为 evaluation v2：JSD 第 3 列为 **CategoryTransition**（类别转移 JSD），非前文旧表的 **Interval**；且 `Category` 口径与旧 two-city-v1 表不同，故本节 JSD **不与前文 JSD 表直接并比**（约束满足指标口径一致、可比）。
- 三种子 135398/135399/135400 的均值 ± 样本标准差（固定检查点采样波动，非多训练不确定性）。
- 类别解码会改变轨迹长度（还原非目录 token 槽位），故并列 **on_matched**（与 off 等长对齐口径）。还原事件数：Istanbul 27/31/30，NewYork 41/30/42（按种子）。
- 极少数退化槽位走 catalog-POI fallback，故 Istanbul 的 on 的 `category_poi_mismatch_rate` 为极小非零（≈0.0001）。
- Istanbul 历史基础训练 batch=512、NewYork=64；全局导出器未改；off 与封存产物未触碰。

### 分布相似性（JSD ↓）

#### Istanbul_PO1_OOD
| 组 | Distance ↓ | Radius ↓ | CategoryTransition ↓ | DailyLoc ↓ | Category ↓ | G-RANK ↓ |
|---|---:|---:|---:|---:|---:|---:|
| off (=PCDG) | 0.1314 ± 0.0035 | 0.0516 ± 0.0022 | 0.0162 ± 0.0006 | 0.0368 ± 0.0014 | 0.0446 ± 0.0020 | 0.1820 ± 0.0016 |
| on (方案A) | 0.1539 ± 0.0028 | 0.0892 ± 0.0024 | 0.0238 ± 0.0011 | 0.0297 ± 0.0012 | 0.0423 ± 0.0009 | 0.1813 ± 0.0018 |
| on_matched | 0.1536 ± 0.0027 | 0.0890 ± 0.0025 | 0.0238 ± 0.0010 | 0.0298 ± 0.0013 | 0.0423 ± 0.0008 | 0.1813 ± 0.0017 |

#### NewYork_PO1_OOD
| 组 | Distance ↓ | Radius ↓ | CategoryTransition ↓ | DailyLoc ↓ | Category ↓ | G-RANK ↓ |
|---|---:|---:|---:|---:|---:|---:|
| off (=PCDG) | 0.2210 ± 0.0224 | 0.1265 ± 0.0051 | 0.0132 ± 0.0004 | 0.0351 ± 0.0035 | 0.0173 ± 0.0013 | 0.1942 ± 0.0041 |
| on (方案A) | 0.1989 ± 0.0198 | 0.1681 ± 0.0042 | 0.0182 ± 0.0012 | 0.0335 ± 0.0022 | 0.0178 ± 0.0012 | 0.1914 ± 0.0059 |
| on_matched | 0.2052 ± 0.0217 | 0.1692 ± 0.0035 | 0.0182 ± 0.0012 | 0.0300 ± 0.0028 | 0.0178 ± 0.0012 | 0.1927 ± 0.0064 |

<!-- PLACEHOLDER_CONSTRAINTS -->

### 约束满足（%；OVR_strict / OVR_skip / Unsat ↓，coverage ↑）

#### Istanbul_PO1_OOD
| 组 | OVR_strict ↓ | OVR_skip ↓ | Pair coverage ↑ | Category coverage ↑ | Unsat ↓ |
|---|---:|---:|---:|---:|---:|
| off (=PCDG) | 32.80 ± 0.50 | 9.12 ± 0.10 | 70.99 ± 0.08 | 85.36 ± 0.15 | 36.86 ± 0.37 |
| on (方案A) | **3.72 ± 0.13** | **1.76 ± 0.10** | **94.74 ± 0.16** | **97.81 ± 0.07** | **6.90 ± 0.18** |
| on_matched | 3.97 ± 0.17 | 1.74 ± 0.09 | 94.47 ± 0.19 | 97.65 ± 0.08 | 7.14 ± 0.22 |

#### NewYork_PO1_OOD
| 组 | OVR_strict ↓ | OVR_skip ↓ | Pair coverage ↑ | Category coverage ↑ | Unsat ↓ |
|---|---:|---:|---:|---:|---:|
| off (=PCDG) | 38.51 ± 0.95 | 14.11 ± 1.22 | 67.60 ± 1.86 | 81.54 ± 0.70 | 42.42 ± 1.84 |
| on (方案A) | **9.52 ± 1.04** | **3.47 ± 0.64** | **88.67 ± 1.74** | **93.92 ± 0.68** | **14.09 ± 2.11** |
| on_matched | 10.19 ± 1.27 | 2.99 ± 0.39 | 88.11 ± 1.89 | 93.36 ± 0.85 | 14.49 ± 2.22 |

### 小结
- 类别一致解码（on）在两城使**约束满足大幅改善**：OVR_strict 由 32.80 / 38.51% 降至 **3.72 / 9.52%**，Pair / Category coverage 显著上升，Unsat 大幅下降，类别-POI 不一致≈归零；三种子一致。
- **分布保真有得有失**：Radius 两城变差，Distance NewYork 略降、Istanbul 略升，CategoryTransition 略升，G-RANK 基本持平，DailyLoc 改善。总体以小幅分布代价换大幅约束收益。
- on 与 on_matched 很接近（还原事件占比小），两种口径结论一致。


# Z2Tensors 接口更改

## Sector 固化：仅支持 Z2Irrep

- 包中只保留 `Z2Irrep` 一个 sector 类型（具体 `struct`，字段 `n::UInt8`），删除了 `ZNIrrep{N}`、`ProductSector` 等多余 sector 定义。
- 所有以 sector 为参数的泛型类型全部固化为具体类型参数：
  - `FusionTree{N}`（原 `FusionTree{I,N}`）
  - `FusionBlockStructure{N₁,N₂}`（`N = N₁ + N₂` 由内部构造算出）
  - `Z2Space`（原 `GradedSpace{I,D}`；dual 改为字段，单类型）
  - `ProductSpace{N}`、`HomSpace{N₁,N₂}`
  - `TensorMap{T,N₁,N₂,A,FBS}`，其中 `FBS<:FusionBlockStructure{N₁,N₂}`
  - `AdjointTensorMap{T,N₁,N₂,TT}`、`DiagonalTensorMap{T,A}`
  - `SectorDict{V} = SortedVectorDict{Z2Irrep,V}`
- 导出列表移除 `ZNIrrep`、`ProductSector`、`AbstractIrrep`、`Irrep`、`GradedSpace`；保留 abstract type `Sector` 仅用于方法签名。

## Sector 融合接口：移除 `⊗()` 空输入行为，改用 `couple`

- `sectors.jl` 中不再提供 `⊗()`（零参）方法；`⊗` 仅保留 sector 的二元/多元融合（`Nsymbol`、空间 `fuse`、张量外积等仍在使用）以及空间上的 `⊗`（`ProductSpace` 构造）。
- 新增 `couple(uncoupled::NTuple{N,Z2Irrep})`，行为与 `⊗(uncoupled...)` 一致，且包含空元组情形（返回 `one(Z2Irrep)`）。
- 所有对（可能为空的）sector 序列求耦合 charge 的调用点统一改用 `couple`，不再手写 `isempty ? one(...) : ⊗(...)` 判断：
  - 融合树置换（`fusiontrees/manipulations.jl` 的 `permute`）
  - `TensorMap` 的 sector 索引 `getindex(t, sectors::Tuple{I,Vararg{I}})`（`tensors/tensor.jl`）
  - `fusiontrees(uncoupled, coupled)` 的合法性检查（`fusiontrees/iterator.jl`）

## TensorMap 结构直接存储，移除 LRU 缓存

- `TensorMap` 新增 `structure::FusionBlockStructure` 字段，由空间直接计算并存储，不再使用 `LRUCache` 缓存；整个包对 LRUCache 的依赖已移除（包括 tree-transformer 的 LRU 缓存）。

## 线性代数：改为基于 MatrixAlgebraKit 的薄封装

- `auxiliary/linalg.jl` 中的矩阵层分解改为调用 MatrixAlgebraKit（0.6.x）：
  - `leftorth!` / `rightorth!`：`MAK.left_orth!(A; alg=:qr/:svd/:polar, positive=..., trunc=trunctol(atol=...))`
  - `_svd!`：`MAK.svd_compact!`（奇异值从 `S.diag` 提取）
- 对外 API 保留原有的算法类型 `QR`、`QRpos`、`LQ`、`LQpos`、`SVD`、`SDD`、`Polar`。

## @tensor 宏支持

- 基于 TensorOperations 5.x 接口（`tensoralloc` / `tensoradd!` / `tensorcontract!` 重写）：
  - 支持 `=` 与 `+=`
  - 支持多张量（3–4 个）收缩
  - 支持通过右端括号改变收缩顺序
  - 支持伴随张量、`conj`、标量系数、标量输出

## 截断方案接口调整

- `TruncationDimCutoff` 改名为 `TruncateDimCutoff`；便捷构造函数 `truncdimcutoff` 保留不变。
- 新增 `TruncateRelError`（字段 `ϵ::Float64`、`add_back::Int`）与构造函数 `truncrelerr(ϵ[, add_back])` / `truncrelerr(; ϵ, add_back)`：
  - 行为类似 `TruncationCutoff`（`truncbelow`），但使用相对截断：先把奇异值矢量按其 p-范数归一化，再丢弃小于 `ϵ` 的奇异值；
  - 若剩余奇异值个数少于 `add_back`，则保留 `add_back` 个（即相对版 `TruncateDimCutoff`，去掉了固定 `D` 截断环节）。
- 两者均已导出。

## 文件重组与纯数组张量分解工具

- `src/auxiliary/auxiliary.jl` 改名为 `src/auxiliary/misc.jl`；主模块文件中所有 `auxiliary` 的 include 统一前置（紧随 `sectors.jl`）。
- 截断方案类型（`TruncationScheme` 及 `NoTruncation`、`TruncationError`、`TruncationDimension`、`TruncationCutoff`、`TruncateDimCutoff`、`TruncateRelError`）移至 `src/auxiliary/tensorfactorizations.jl`；`TruncationSpace` 与 sector 级截断机器（`_compute_truncdim`/`_compute_truncerr`/`_findnext*`）留在 `src/tensors/truncation.jl`。
- 新增 `src/auxiliary/tensorfactorizations.jl`：移植 TEMPO（`src/tensorops/tensorfactorizations.jl`，矩阵级 `_truncate!` 取自其 `truncation.jl`）的纯数组/稠密张量分解工具：
  - `tsvd!` / `tsvd`（矩阵级与 left/right 分组的稠密张量级，`alg=SDD()/SVD()` 对应 MAK `SafeDivideAndConquer`/`QRIteration`）
  - `leftorth!` / `leftorth` / `rightorth!` / `rightorth`（矩阵级关键字版与稠密张量分组版，委托给已有的矩阵级位置参数实现）
  - 截断方案对普通奇异值矢量的 `_truncate!`（`NoTruncation`、`truncdim`、`truncerr`、`truncbelow`、`truncrelerr`、`truncdimcutoff`；`truncdimcutoff` 返回相对误差，其余返回绝对尾部范数，与 TEMPO 一致）
  - 辅助工具：数组 `permute` 视图、`tie`、`isometry`、`renyi_entropy`
- 导出调整：新增导出 `TruncationScheme`、`NoTruncation`、`tie`、`isometry`、`renyi_entropy`；`truncdim` 新增关键字构造 `truncdim(; D)`。`QR`/`QRpos`/`LQ`/`LQpos`/`SVD`/`SDD`/`Polar` 维持原导出不变。
- 测试新增 `test/tensorfactorizations.jl`（参照 TEMPO `test/api/truncation.jl` 与 `linalg.jl`；`add_back > D` 按本包语义改为抛 `ArgumentError`）。

## 其他

- 空空间构造新增 `Z2Space(; dual)` 方法（factorizations 内 `S(dims)` 空 dict splat 需要）。
- `SectorDict(kv)` 泛型构造函数加入函数屏障（`_SectorDict`），保证类型稳定（tsvd truncation error 的 `@constinferred` 依赖此点）。
- 已知约定（非 bug）：`@tensor` 中 `conj(A[...])` 收缩结果空间带 dual 标记（如 `W2'←W2'`），与 TensorKit 行为一致。

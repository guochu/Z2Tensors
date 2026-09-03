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

## 其他

- 空空间构造新增 `Z2Space(; dual)` 方法（factorizations 内 `S(dims)` 空 dict splat 需要）。
- `SectorDict(kv)` 泛型构造函数加入函数屏障（`_SectorDict`），保证类型稳定（tsvd truncation error 的 `@constinferred` 依赖此点）。
- 已知约定（非 bug）：`@tensor` 中 `conj(A[...])` 收缩结果空间带 dual 标记（如 `W2'←W2'`），与 TensorKit 行为一致。

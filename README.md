# 联邦迁移学习故障诊断系统 (Federated Transfer Learning - Fault Diagnosis)

一个用于不同工厂设备间故障诊断模型迁移的联邦迁移学习系统。

## 系统架构

### 核心模块

1. **联邦协调器 (Federated Coordinator)** - 服务器端核心组件
   - 管理多个客户端
   - 执行FedAvg参数聚合
   - 协调全局训练流程

2. **客户端训练器 (Client Trainer)** - 本地训练模块
   - 每个客户端独立训练CNN模型
   - 提取/上传特征提取器参数
   - 接收并加载全局模型

3. **DANN域对抗网络 (Domain Adversarial Neural Network)**
   - 梯度反转层 (Gradient Reversal Layer)
   - 域判别器对齐特征分布
   - 实现跨域迁移学习

4. **模型仓库 (Model Repository)**
   - 模型版本管理
   - 训练日志记录
   - 模型保存/加载

5. **监控面板前端 (Monitoring Dashboard)**
   - 实时训练状态监控
   - 客户端管理
   - 模型管理
   - 故障诊断预测

## 技术栈

### 后端
- **Python 3.8+**
- **PyTorch** - 深度学习框架
- **FastAPI** - Web API框架
- **NumPy/SciPy** - 数据处理

### 前端
- **React 18** + **TypeScript**
- **Vite** - 构建工具
- **Tailwind CSS** - 样式框架
- **Recharts** - 图表库
- **Lucide React** - 图标库

## 快速开始

### 1. 安装后端依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 运行后端服务

```bash
cd backend
python main.py
```

API文档: http://localhost:8000/docs

### 3. 安装前端依赖

```bash
cd frontend
npm install
```

### 4. 运行前端服务

```bash
cd frontend
npm run dev
```

前端地址: http://localhost:3000

### 5. 运行演示脚本

```bash
cd backend
python run_demo.py
```

## API接口

### 系统状态
- `GET /api/v1/status` - 获取系统状态
- `GET /api/v1/clients` - 获取客户端列表
- `GET /api/v1/datasets/{client_id}` - 获取数据集详情

### 训练控制
- `POST /api/v1/training/start` - 开始联邦训练
- `GET /api/v1/training/status` - 获取训练状态
- `GET /api/v1/training/sessions` - 获取训练会话列表

### 模型管理
- `GET /api/v1/models` - 获取模型列表
- `GET /api/v1/models/{version_id}` - 获取模型详情
- `DELETE /api/v1/models/{version_id}` - 删除模型

### 预测
- `POST /api/v1/predict` - 故障诊断预测
- `GET /api/v1/fault-types` - 获取故障类型列表

## 故障类型

系统支持5种故障类型诊断：
0. **Normal** - 正常运行
1. **Bearing Fault** - 轴承故障
2. **Gear Fault** - 齿轮故障
3. **Unbalance** - 不平衡
4. **Misalignment** - 不对中

## 模拟工况

系统模拟4个不同工厂的工况：
- **factory_A** - 标准工况
- **factory_B** - 高速低载
- **factory_C** - 低速重载
- **factory_D** - 高噪声环境

## 项目结构

```
fed-transfer-diag-8/
├── backend/
│   ├── app/
│   │   ├── api/              # API路由
│   │   ├── core/             # 核心模块（数据模拟）
│   │   ├── models/           # 模型定义（CNN、DANN）
│   │   ├── services/         # 服务层（训练器、协调器）
│   │   └── schemas/          # Pydantic数据结构
│   ├── main.py               # FastAPI入口
│   ├── run_demo.py           # 演示脚本
│   └── requirements.txt      # Python依赖
├── frontend/
│   ├── src/
│   │   ├── components/       # React组件
│   │   ├── pages/            # 页面组件
│   │   └── services/         # API服务
│   └── package.json          # Node.js依赖
├── models/                    # 模型存储
├── logs/                      # 训练日志
└── data/                      # 数据存储
```

## 联邦学习流程

1. **初始化** - 服务器初始化全局模型
2. **广播** - 全局模型发送到所有客户端
3. **本地训练** - 每个客户端用本地数据训练
4. **参数上传** - 客户端上传特征提取器参数
5. **联邦聚合** - 服务器执行FedAvg聚合
6. **域对齐** - 定期执行DANN域对抗训练
7. **模型更新** - 广播更新后的全局模型
8. **重复** - 重复步骤3-7直到收敛

## License

MIT

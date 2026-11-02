# build/

可复现的构建入口与实现报告。三个库的构建产物分开存放：

| 子目录 | 内容 |
|---|---|
| cnn/ | 编译器/IP 打包 Tcl 与 PS 脚本、历史 r4 与旧全算子平台的利用率/时序报告 |
| snn/ | compile/build/package 生成的 manifest 与调度、OOC 与平台 r7 的时序/资源报告 |
| lnn/ | 仿真入口（run_sim.tcl）与验收报告 |

构建脚本按"编译 → 仿真门禁 → OOC → 平台实现 → 位流"执行；报告只保留文本（.rpt/.txt/.json），
不复制 Vivado 工程与临时目录。

// ================================================================
// Typst 实验报告 - 虚拟仪器
// ================================================================

// --- 1. 文档与页面设置 ---
#set document(
  title: "虚拟仪器实验报告",
  author: "[REDACTED]",
)

#let info-line(label, content) = {
  // 使用 block 创建一个带有底部边框的块，效果等同于下划线
  let underlined-content = block(
    width: 20%, // 宽度撑满
    stroke: (bottom: 0.5pt), // 只在底部画线
    inset: (bottom: 5pt), // 文本与线的间距
    align(center, content), // 内容水平居中
  )

  // 使用 grid 布局，让标签和下划线内容对齐
  grid(
    columns: (auto, 1fr),
    label, underlined-content,
  )
}

#set page(
  paper: "a4",
  margin: (top: 1in, bottom: 1in, left: 1in, right: 1in),
  header: [
    #grid(
      columns: (1fr, auto, 1fr),
      [#text(9pt)[实验名称：虚拟仪器实验]], align(right)[#text(9pt)[普通物理实验报告]],
    )
    #v(-0.5em)
    #line(length: 100%, stroke: 0.5pt)
  ],
  footer: context [
    #align(center)[
      #text(10pt)[
        第 #counter(page).display() 页 / 共 #counter(page).display(both: true).at(1) 页
      ]
    ]
  ],
)

// --- 2. 字体与段落设置 ---
#set text(
  font: "Songti SC",
  size: 12pt,
)
#set par(
  first-line-indent: 2em,
)


// --- 3. 自定义函数 ---
#let question(body) = {
  set text(font: "Kaiti SC", weight: "bold")
  body
}

// ================================================================
// 正文内容
// ================================================================

#align(center)[
  #v(2em)
  #text(1.5em, weight: "bold")[虚拟仪器实验报告]
  #v(0.5em)
  #text(1.2em, weight: "bold")[普通物理实验2]
  #v(1em)

  #grid(
    columns: (auto, 3cm),
    column-gutter: 1em,
    row-gutter: 0.6em,
    "姓名：",
    block(
      width: 100%,
      stroke: (bottom: 0.5pt),
      inset: (bottom: 4pt),
      align(center)[[REDACTED]],
    ),

    "学号：",
    block(
      width: 100%,
      stroke: (bottom: 0.5pt),
      inset: (bottom: 4pt),
      align(center)[[REDACTED]],
    ),

    "Week：",
    block(
      width: 100%,
      stroke: (bottom: 0.5pt),
      inset: (bottom: 4pt),
      align(center)[01 & 02],
    ),
  )
  #v(2em)
]

= 第一周：虚拟仪器基础测量

== 1. 电阻测量 ($50 Omega$ 与 $1k Omega$)
#v(0.5em)
利用搭建的虚拟仪器程序对标称值为 50$Omega$ 和 1k$Omega$ 的电阻进行了测量。

#figure(
  image("data/week 01/50ohm-1.jpg", width: 80%),
  caption: [50$Omega$ 电阻测量程序前面板截图],
)
#v(1em)

测量数据记录如下表所示：

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    inset: 10pt,
    align: center,
    [*电阻标称值*], [*测量值 1 ($Omega$)*], [*测量值 2 ($Omega$)*], [*测量值 3 ($Omega$)*],
    [50 $Omega$], [], [], [],
    [1k $Omega$], [], [], [],
  ),
  caption: [电阻测量数据记录表],
)

== 2. 二极管伏安曲线测量

利用虚拟仪器测量了二极管的正向和反向伏安特性曲线。

#figure(
  image("data/week 01/UI diodode diagram.jpg", width: 80%),
  caption: [二极管伏安特性测量程序框图/前面板],
)

*(选做) 静态电阻计算：*
根据测量结果，在电流 $I = plus.minus 4$ mA 处计算二极管的静态电阻 $R = U/I$。

- 正向 4mA 处：$U approx$ ____ V, $R_{+} = U / (4 "mA") approx$ ____ $Omega$。
- 反向 -4mA 处：$U approx$ ____ V, $R_{-} = U / (-4 "mA") approx$ ____ $Omega$。

#pagebreak()

= 第二周：Fano 共振实验

== 1. 实验电路与原理

本实验旨在研究 Fano 共振现象。Fano 共振是一种量子干涉效应，表现为分立态与连续态之间的干涉，导致光谱呈现不对称的线型。在宏观电路中，可以通过耦合振荡回路来模拟这一现象。

实验电路主要由信号源、耦合电容 $C$ 以及并联谐振回路（由电感 $L$、谐振电容 $C_2$ 和损耗电阻 $R$ 组成）构成。

// 简单的电路示意图绘制
#figure(
  box(stroke: 1pt, inset: 20pt, radius: 5pt)[
    #place(center)[
      // 信号源 -> C -> Node -> (L || C2 || R) -> GND
      #set align(center)
      #block(height: 150pt, width: 100%)[
        // 信号源
        #place(dx: 20pt, dy: 60pt)[$V_{in}$]
        #place(dx: 40pt, dy: 70pt)[#circle(radius: 10pt, stroke: 1pt)]
        #place(dx: 50pt, dy: 70pt)[#line(length: 40pt, stroke: 1pt)] // 连线

        // 耦合电容 C
        #place(dx: 90pt, dy: 60pt)[$C$]
        #place(dx: 90pt, dy: 65pt)[#line(start: (0pt, 0pt), end: (0pt, 10pt), stroke: 1pt)]
        #place(dx: 95pt, dy: 65pt)[#line(start: (0pt, 0pt), end: (0pt, 10pt), stroke: 1pt)]
        #place(dx: 95pt, dy: 70pt)[#line(length: 40pt, stroke: 1pt)] // 连线到节点

        // 节点
        #place(dx: 135pt, dy: 70pt)[#circle(radius: 2pt, fill: black)]

        // 并联支路向下
        #place(dx: 135pt, dy: 70pt)[#line(start: (0pt, 0pt), end: (0pt, 40pt), stroke: 1pt)]
        #place(dx: 135pt, dy: 110pt)[#line(start: (-60pt, 0pt), end: (60pt, 0pt), stroke: 1pt)] // 横线分流

        // L 支路
        #place(dx: 75pt, dy: 110pt)[#line(start: (0pt, 0pt), end: (0pt, 20pt), stroke: 1pt)]
        #place(dx: 75pt, dy: 130pt)[$L$]
        #place(dx: 75pt, dy: 140pt)[#rect(width: 10pt, height: 20pt, stroke: 1pt)]
        #place(dx: 75pt, dy: 160pt)[#line(start: (0pt, 0pt), end: (0pt, 20pt), stroke: 1pt)]

        // C2 支路
        #place(dx: 135pt, dy: 110pt)[#line(start: (0pt, 0pt), end: (0pt, 20pt), stroke: 1pt)]
        #place(dx: 135pt, dy: 130pt)[$C_2$]
        #place(dx: 130pt, dy: 140pt)[#line(start: (0pt, 0pt), end: (10pt, 0pt), stroke: 1pt)]
        #place(dx: 130pt, dy: 145pt)[#line(start: (0pt, 0pt), end: (10pt, 0pt), stroke: 1pt)]
        #place(dx: 135pt, dy: 145pt)[#line(start: (0pt, 0pt), end: (0pt, 35pt), stroke: 1pt)]

        // R 支路
        #place(dx: 195pt, dy: 110pt)[#line(start: (0pt, 0pt), end: (0pt, 20pt), stroke: 1pt)]
        #place(dx: 195pt, dy: 130pt)[$R$]
        #place(dx: 190pt, dy: 140pt)[#rect(width: 10pt, height: 20pt, stroke: 1pt)]
        #place(dx: 195pt, dy: 160pt)[#line(start: (0pt, 0pt), end: (0pt, 20pt), stroke: 1pt)]

        // 汇合接地
        #place(dx: 135pt, dy: 180pt)[#line(start: (-60pt, 0pt), end: (60pt, 0pt), stroke: 1pt)]
        #place(dx: 135pt, dy: 180pt)[#line(start: (0pt, 0pt), end: (0pt, 10pt), stroke: 1pt)]
        #place(dx: 135pt, dy: 190pt)[GND]
      ]
    ]
  ],
  caption: [Fano 共振实验电路示意图],
)

Fano 共振的线型公式通常表示为：
$ sigma(epsilon) = (q + epsilon)^2 / (1 + epsilon^2) $
其中，$epsilon = 2(E - E_0) / Gamma$ 为归一化能量（频率），$E_0$ 为共振中心能量，$Gamma$ 为共振宽度，$q$ 为 Fano 非对称因子。$q$ 的大小决定了曲线的不对称程度：
- 当 $|q| -> infinity$ 时，表现为洛伦兹线型（对称）。
- 当 $q = 0$ 时，表现为反共振（凹陷）。
- 当 $q approx 1$ 时，表现为典型的非对称 Fano 线型。

== 2. 元件表征值测量

实验首先对 Fano 共振电路中使用的主要元件（电感 L 和电容 C）进行了频率特性表征。测量结果如下图所示：

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(image("report_images/component_16mH.png", width: 100%), caption: [L=16mH 电感测量结果]),
  figure(image("report_images/component_18mH.png", width: 100%), caption: [L=18mH 电感测量结果]),
)
#figure(
  image("report_images/component_047uF.png", width: 60%),
  caption: [C=0.047$mu "F"$ 电容测量结果],
)

== 3. Fano 共振幅频特性曲线研究

通过改变 Fano 共振电路中的参数（耦合电容 $C$、谐振电容 $C_2$、损耗电阻 $R$），测量了多组幅频特性曲线。为了清晰展示参数变化对共振曲线的影响，我们将同一元件不同参数下的曲线绘制在同一张图中。图中*浅色实线*为 Spline 拟合曲线，*深色散点*为实验测量数据。

=== (1) 改变谐振电容 $C_2$ 的影响

保持耦合电容 $C = 0.5 mu "F"$ 和 $R = 500 Omega$ 不变，改变谐振回路电容 $C_2$ 的值。

#figure(
  image("report_images/fano_comparison_vary_c2.png", width: 90%),
  caption: [不同谐振电容 $C_2$ 下的 Fano 共振幅频曲线],
)

*分析与讨论：*
$C_2$ 是并联谐振回路的核心元件之一。根据 $L C$ 振荡原理，谐振频率 $f_0 approx 1 / (2 pi sqrt(L C_2))$。
从图中可以清晰地观察到，随着 $C_2$ 的增大（从 $0.03 mu "F"$ 到 $0.3 mu "F"$），共振峰的位置显著向低频方向移动。这与公式预测完全一致。
同时，共振峰的宽度和高度也随之改变，这反映了电路品质因数 $Q$ 的变化。在 $C_2$ 较小时，共振频率较高，电路的特征阻抗 $sqrt(L/C_2)$ 较大，可能导致更尖锐的共振峰。

=== (2) 改变耦合电容 $C$ 的影响

保持谐振电容 $C_2 = 0.2 mu "F"$ 和 $R = 500 Omega$ 不变，改变耦合电容 $C$ 的值。

#figure(
  image("report_images/fano_comparison_vary_c.png", width: 90%),
  caption: [不同耦合电容 $C$ 下的 Fano 共振幅频曲线],
)

*分析与讨论：*
电容 $C$ 在电路中起耦合作用，连接信号源与谐振回路。它决定了“连续态”（背景直通信号）与“分立态”（谐振回路信号）之间的干涉强度，即直接影响 Fano 公式中的 $q$ 因子。
从图中可以看出，随着 $C$ 的变化，共振曲线的形状发生显著变化，不对称性（Fano profile）变得更加明显或趋于平缓。
- 较小的 $C$ 意味着弱耦合，背景通道信号较弱，共振主要由 $L C$ 回路主导，线型可能更接近对称的洛伦兹型（$|q|$ 较大）。
- 较大的 $C$ 增强了直通信号，使得干涉效应增强，导致线型出现明显的非对称谷-峰结构。

=== (3) 改变电阻 $R$ 的影响

保持 $C = 0.5 mu "F"$ 和 $C_2 = 0.2 mu "F"$ 不变，改变电阻 $R$ 的值。

#figure(
  image("report_images/fano_comparison_vary_r.png", width: 90%),
  caption: [不同电阻 $R$ 下的 Fano 共振幅频曲线],
)

*分析与讨论：*
电阻 $R$ 引入了电路的损耗（阻尼）。它直接关联到共振宽度 $Gamma$。
- 当 $R$ 较小（假设为串联损耗）或较大（假设为并联损耗，视具体电路连接而定，通常并联电阻越小阻尼越大）时，共振峰会被展宽，幅度降低。
- 从图中可以看到，随着阻值的变化，共振峰的锐度（Sharpness）发生明显变化。高阻尼状态下，共振现象被抑制，曲线趋于平坦；低阻尼状态下，共振峰尖锐且高耸。
- 此外，损耗也会影响相位变化，进而微调 Fano 干涉的形态。

== 4. 总结

本次实验通过虚拟仪器成功复现了 Fano 共振现象。实验结果表明：
1. 谐振电容 $C_2$ 主要决定共振中心频率 $f_0$。
2. 耦合电容 $C$ 主要调节耦合强度和 Fano 非对称因子 $q$。
3. 电阻 $R$ 主要控制共振的阻尼和宽度 $Gamma$。
实验曲线与 Fano 理论公式定性吻合，验证了该电路模型在模拟量子干涉效应方面的有效性。

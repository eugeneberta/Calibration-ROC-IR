using Random,
      Plots,
      PlotlyJS,
      Measures,
      LaTeXStrings,
      Polyhedra,
      GLPK,
      NPZ
include("utils.jl")

solver = GLPK.Optimizer
lib = DefaultLibrary{Float64}(solver)


#####################
### 2D Experiment ###
#####################
Pcal = npzread("predictions/Cover2Pcal.npy")
Ycal = npzread("predictions/Cover2Ycal.npy")
Ptest = npzread("predictions/Cover2Ptest.npy")
Ytest = npzread("predictions/Cover2Ytest.npy")

Δg = vcat([[i 1000-i]; for i in 0:1000]...) ./ 1000 # Binary simplex grid.
Rcal, Rgcal, IRPnregions, IRPcalCE, IRPtestCE, IRPcalAUC, IRPtestAUC = MulticlassIRP(Pcal, Ycal, Δg, true, Ptest, Ytest)
Bcal, Bgcal, BINnregions, BINcalCE, BINtestCE, BINcalAUC, BINtestAUC = MulticlassBinning(Pcal, Ycal, Δg, 4, true, Ptest, Ytest)

calInitCE = regcrossentropy(Pcal, Ycal)
testInitCE = crossentropy(Ptest, Ytest)
calInitAUC = rocauc(Pcal, Ycal)
testInitAUC = rocauc(Ptest, Ytest)

gr()
p1 = Plots.plot(BINnregions, BINcalAUC,
    label="Recursive binning",
    marker=:dot,
    linewidth=3,
    title="K=2, Calibration AUC",
    xaxis="Number of bins",
    yaxis="AUC"
)
Plots.plot!(IRPnregions, IRPcalAUC,
    label="IRP",
    marker=:dot,
    linewidth=3
)
Plots.hline!([calInitAUC], label="AUC before calibration", linewidth=3)

p2 = Plots.plot(BINnregions, BINtestAUC,
label="Recursive binning",
marker=:dot,
linewidth=3,
title="K=2, Test AUC",
xaxis="Number of bins",
yaxis="AUC",
legend=true
)
Plots.plot!(IRPnregions, IRPtestAUC,
    label="IRP",
    marker=:dot,
    linewidth=3
)
Plots.hline!([testInitAUC], label="AUC before Calibration", linewidth=3)

p3 = Plots.plot(BINnregions, BINcalCE,
label="Recursive binning",
marker=:dot,
linewidth=3,
title="K=2, Calibration Cross Entropy",
xaxis="Number of bins",
yaxis="regularized CE"
)
Plots.plot!(IRPnregions, IRPcalCE,
    label="IRP",
    marker=:dot,
    linewidth=3
)
Plots.hline!([calInitCE], label="CE before calibration", linewidth=3)

p4 = Plots.plot(BINnregions, BINtestCE,
label="Recursive binning",
marker=:dot,
linewidth=3,
title="K=2, Test Cross Entropy",
xaxis="Number of bins",
yaxis="CE",
legend=true
)
Plots.plot!(IRPnregions, IRPtestCE,
    label="IRP",
    marker=:dot,
    linewidth=3,
)
Plots.hline!([testInitCE], label="CE before Calibration", linewidth=3)

p = Plots.plot(
    p1, p2, p3, p4,
    layout=(2,2),
    size=(800, 800),
    left_margin=[3mm -5mm],
    right_margin=[0mm 1mm],
    bottom_margin=5mm,
    titlelocation=:center,
    xlabelfontsize=10, ylabelfontsize=10
)
Plots.savefig(p, "figures/2DIRPvsBinning.pdf")

# ROC Curve
g = vcat([[i 1000-i]; for i in -1:1001]...) ./ 1000
ROCP = rocsurface(Pcal, Ycal, g)
ROCR = rocsurface(Rcal, Ycal, g)

p = Plots.scatter(ROCP[:,1], ROCP[:,2], markersize=2, size=(500,400), label="Initial forecasts", leg=:bottomleft)
Plots.scatter!(ROCR[:,1], ROCR[:,2], markersize=3, label="Calibrated forecasts")
Plots.title!("2D ROC surface")
Plots.xlabel!(L"p_1(\gamma)")
Plots.ylabel!(L"p_2(\gamma)")
Plots.savefig(p, "figures/2DROCsurface.pdf")


#####################
### 3D Experiment ###
#####################
Pcal = npzread("predictions/Cover3Pcal.npy")
Ycal = npzread("predictions/Cover3Ycal.npy")
Ptest = npzread("predictions/Cover3Ptest.npy")
Ytest = npzread("predictions/Cover3Ytest.npy")

Δg = vcat([[i j 100-i-j]; for i in 0:100 for j in 0:100 if i+j<=100]...) ./ 100
Rcal, Rgcal, IRPnregions, IRPcalCE, IRPtestCE, IRPcalAUC, IRPtestAUC = MulticlassIRP(Pcal, Ycal, Δg, true, Ptest, Ytest)
Bcal, Bgcal, BINnregions, BINcalCE, BINtestCE, BINcalAUC, BINtestAUC = MulticlassBinning(Pcal, Ycal, Δg, 4, true, Ptest, Ytest)

calInitCE = regcrossentropy(Pcal, Ycal)
testInitCE = crossentropy(Ptest, Ytest)
calInitAUC = rocauc(Pcal, Ycal)
testInitAUC = rocauc(Ptest, Ytest)

gr()
p1 = Plots.plot(BINnregions, BINcalAUC,
    label="Recursive binning",
    marker=:dot,
    linewidth=3,
    title="K=3, Calibration VUS",
    xaxis="Number of bins",
    yaxis="VUS"
)
Plots.plot!(IRPnregions, IRPcalAUC,
    label="IRP",
    marker=:dot,
    linewidth=3
)
Plots.hline!([calInitAUC], label="VUS before calibration", linewidth=3)

p2 = Plots.plot(BINnregions, BINtestAUC,
label="Recursive binning",
marker=:dot,
linewidth=3,
title="K=3, Test VUS",
xaxis="Number of bins",
yaxis="VUS",
legend=true
)
Plots.plot!(IRPnregions, IRPtestAUC,
    label="IRP",
    marker=:dot,
    linewidth=3
)
Plots.hline!([testInitAUC], label="VUS before Calibration", linewidth=3)

p3 = Plots.plot(BINnregions, BINcalCE,
label="Recursive binning",
marker=:dot,
linewidth=3,
title="K=3, Calibration Cross Entropy",
xaxis="Number of bins",
yaxis="regularized CE"
)
Plots.plot!(IRPnregions, IRPcalCE,
    label="IRP",
    marker=:dot,
    linewidth=3
)
Plots.hline!([calInitCE], label="CE before calibration", linewidth=3)

p4 = Plots.plot(BINnregions, BINtestCE,
label="Recursive binning",
marker=:dot,
linewidth=3,
title="K=3, Test Cross Entropy",
xaxis="Number of bins",
yaxis="CE",
legend=true
)
Plots.plot!(IRPnregions, IRPtestCE,
    label="IRP",
    marker=:dot,
    linewidth=3,
)
Plots.hline!([testInitCE], label="CE before Calibration", linewidth=3)

p = Plots.plot(
    p1, p2, p3, p4,
    layout=(2,2),
    size=(800, 800),
    left_margin=[3mm -5mm],
    right_margin=[0mm 1mm],
    bottom_margin=5mm,
    titlelocation=:center,
    xlabelfontsize=10, ylabelfontsize=10
)
Plots.savefig(p, "figures/3DIRPvsBinning.pdf")

# Calibration function:
p1 = plotcal3D(Pcal, Ycal, "Initial predictions", :circle, 3, 0.2)
p2 = plotcal3D(Pcal, Rcal, "Calibrated predictions", :circle, 3, 0.2)
p = Plots.plot(p1, p2, layout=(1,2), legend=false, size=(1000, 500))
Plots.savefig(p, "figures/3DCalibrationFunctionsDiscrete.pdf")

p1 = plotcal3D(Δg, Rgcal, "IRP", :circle, 2.6, 0.)
p2 = plotcal3D(Δg, Bgcal, "Binning", :circle, 2.6, 0.)
p = Plots.plot(p1, p2, layout=(1,2), legend=false, size=(1000, 500))
Plots.savefig(p, "figures/3DCalibrationFunctions.pdf")

# ROC surface
g = vcat([[i j k]; for i in -51:151 for j in -51:151 for k in -51:151 if i+j+k==100]...) ./ 100
ROCP = rocsurface(Pcal, Ycal, g)
ROCR = rocsurface(Rcal, Ycal, g)

plotlyjs()
p = Plots.scatter(
    ROCP[:,1], ROCP[:,2], ROCP[:,3],
    markersize=1,
    size=(1000,800),
    legend=true,
    label="Initial ROC surface",
    title="3D ROC surface",
)
Plots.scatter!(
    ROCR[:,1], ROCR[:,2], ROCR[:,3],
    markersize=2,
    label="Calibrated ROC surface"
)
Plots.savefig(p, "figures/3DROCsurface.html")


#####################
### 4D Experiment ###
#####################
Pcal = npzread("predictions/Cover4Pcal.npy")
Ycal = npzread("predictions/Cover4Ycal.npy")
Ptest = npzread("predictions/Cover4Ptest.npy")
Ytest = npzread("predictions/Cover4Ytest.npy")

Δg = vcat([[i j k 50-i-j-k]; for i in 0:50 for j in 0:50 for k in 0:50 if i+j+k<=50]...) ./ 50
Rcal, Rgcal, IRPnregions, IRPcalCE, IRPtestCE, IRPcalAUC, IRPtestAUC = MulticlassIRP(Pcal, Ycal, Δg, true, Ptest, Ytest)
Bcal, Bgcal, BINnregions, BINcalCE, BINtestCE, BINcalAUC, BINtestAUC = MulticlassBinning(Pcal, Ycal, Δg, 4, true, Ptest, Ytest)

calInitCE = regcrossentropy(Pcal, Ycal)
testInitCE = crossentropy(Ptest, Ytest)
calInitAUC = rocauc(Pcal, Ycal)
testInitAUC = rocauc(Ptest, Ytest)

gr()
p1 = Plots.plot(BINnregions, BINcalAUC,
    label="Recursive binning",
    marker=:dot,
    linewidth=3,
    title="K=4, Calibration VUS",
    xaxis="Number of bins",
    yaxis="VUS"
)
Plots.plot!(IRPnregions, IRPcalAUC,
    label="IRP",
    marker=:dot,
    linewidth=3
)
Plots.hline!([calInitAUC], label="VUS before calibration", linewidth=3)

p2 = Plots.plot(BINnregions, BINtestAUC,
label="Recursive binning",
marker=:dot,
linewidth=3,
title="K=4, Test VUS",
xaxis="Number of bins",
yaxis="VUS",
legend=true
)
Plots.plot!(IRPnregions, IRPtestAUC,
    label="IRP",
    marker=:dot,
    linewidth=3
)
Plots.hline!([testInitAUC], label="VUS before Calibration", linewidth=3)

p3 = Plots.plot(BINnregions, BINcalCE,
label="Recursive binning",
marker=:dot,
linewidth=3,
title="K=4, Calibration Cross Entropy",
xaxis="Number of bins",
yaxis="regularized CE"
)
Plots.plot!(IRPnregions, IRPcalCE,
    label="IRP",
    marker=:dot,
    linewidth=3
)
Plots.hline!([calInitCE], label="CE before calibration", linewidth=3)

p4 = Plots.plot(BINnregions, BINtestCE,
label="Recursive binning",
marker=:dot,
linewidth=3,
title="K=4, Test Cross Entropy",
xaxis="Number of bins",
yaxis="CE",
legend=true
)
Plots.plot!(IRPnregions, IRPtestCE,
    label="IRP",
    marker=:dot,
    linewidth=3,
)
Plots.hline!([testInitCE], label="CE before Calibration", linewidth=3)

p = Plots.plot(
    p1, p2, p3, p4,
    layout=(2,2),
    size=(800, 800),
    left_margin=[3mm -5mm],
    right_margin=[0mm 1mm],
    bottom_margin=5mm,
    titlelocation=:center,
    xlabelfontsize=10, ylabelfontsize=10
)
Plots.savefig(p, "figures/4DIRPvsBinning.pdf")

### Calibration set ###
# Plotting calibration function:
plotlyjs()
p0 = plotcal4D(Pcal, Ycal, "Initial predictions", :circle, 1.5)
p1 = plotcal4D(Pcal, Rcal, "Calibrated predictions", :circle, 1.5)
p = Plots.plot(p0, p1, layout=(1,2), legend=false, size=(1000, 500))
Plots.savefig(p, "figures/4DCalibrationFunctions.html")

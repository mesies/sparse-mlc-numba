import MlcLinReg
import helpers

# for i in range(100, 100):
#     helpers.plot_roc_curve(MlcLinReg.MlcLinReg(batch_size=i), savefig=True)
helpers.plot_roc_curve(MlcLinReg.MlcLinReg(batch_size=256), dataset=helpers.load_delicious(2), savefig=False)

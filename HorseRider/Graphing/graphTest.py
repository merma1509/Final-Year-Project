from graphing import Graphing
import numpy as np

grph = Graphing()

def testForRowAndColumns(valueList):
    for nbr in valueList:
        result = grph.findPositionAndSize(nbr)

        if ((result[0]**2) > (result[0]*result[1])):
            print("nbr= ",str(nbr)," res= ", result, "not passed")
            return False
        elif ((result[0]*result[1]) < nbr):
            print("nbr= ",str(nbr)," res= ", result, "not passed")
            return False
        elif ((result[0]**2 > nbr and result[0]**2 != result[0]*result[1])):
            print("nbr= ",str(nbr)," res= ", result, "not passed")
            return False
        print("nbr= ",str(nbr)," res= ", result, "passed")
def testForGraphDrawing():
    x = np.linspace(0, 20)
    y1 = x + 2
    y2 = x**2 + x + 3
    y3 = x**3 + x**2 + x + 4
    grph.NewGraph("linear", x, y1)
    grph.NewGraph("quadratic", x, y2)
    grph.NewGraph("cubic", x, y3)

    grph.NewGraph("liner", x, y1)
    grph.NewGraph("quadr", x, y2)
    grph.NewGraph("cub", x, y3)
    
    grph.NewGraph("linr", x, y1)
    grph.NewGraph("qudr", x, y2)
    grph.NewGraph("cb", x, y3)

    
    grph.NewGraph("iner", x, y1)
    grph.NewGraph("uadr", x, y2)
    grph.NewGraph("ub", x, y3)

    
    grph.NewGraph("lier", x, y1)
    grph.NewGraph("qudr", x, y2)
    grph.NewGraph("cu", x, y3)

    grph.graphing(["linear","quadratic", "cubic","liner","quadr", "cub", "linr", "qudr", "cb", "iner", "uadr", "ub", "lier", "qudr", "cu"], 15)
    print(y1, "\n", y2, "\n", y3)
testForGraphDrawing()
testForRowAndColumns([16, 23, 12, 34, 124, 15, 66, 43, 58])
    


            
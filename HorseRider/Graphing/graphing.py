import matplotlib.pyplot as plt
import numpy as np

class Graphing:
    def __init__(self):
        self.x = []
        self.y = []
        self.graphMapping = {}
        self.nbrGraph = 0
    
    def graphProperties(self, title, labelx, labely, color=None, flipy=False):
        GraphProperties ={ "properties"+str(title): {"color":color, "title":title, "labelx":labelx, "labely":labely, "flipy":flipy}}
        self.graphMapping.update(GraphProperties)

    def NewGraph(self, title, dataX, dataY, labelx="x", labely="y", color="red", flipy=True):
        GraphMap = {str(title): self.nbrGraph}
        self.nbrGraph += 1
        self.graphMapping.update(GraphMap)   
        self.graphProperties(title, labelx, labely, color, flipy)
        self.x.append(dataX)
        self.y.append(dataY)
    
    # this function searches from the stored graph all the graphs
    # that are needed to be graphed and through name we graph it 
    def findPositionAndSize(self, n):
        rows = int(np.sqrt(n))
        if rows <= 3:
            if(n > (rows*rows)):
                columns = rows + 1
            elif(n==(rows*rows)):
                columns = rows
        elif rows > 3:
            rows, columns = 3, 3

        if(n<9):
            return [1, [rows, columns]]
        else:
            return [0, (n-9), [rows, columns]] # if zero is returned then it means we need to create two plot images

    def subPlots(self, rows, columns, graphNameList, nGraphs):
        fig, ax = plt.subplots(rows, columns)
        nbrCol, nbrRow = 0, 0
        for i in range(len(graphNameList)):
            #print(self.graphMapping)
            print(nGraphs, i)
            graphNbr = graphNameList[i]                # getting the name from the list of names
            graphIndex = self.graphMapping[graphNbr]   # using the name as the key in our dictionary
            dataX = self.x[graphIndex]              # using the index to access stored data in x and y data of our class
            dataY = self.y[graphIndex]
            graphProperties = self.graphMapping["properties"+str(graphNbr)]
            if(rows>1):
                ax[nbrRow, nbrCol].plot(dataX, dataY, graphProperties["color"])
                ax[nbrRow, nbrCol].set_title(graphProperties["title"])
                ax[nbrRow, nbrCol].set_ylabel(graphProperties["labely"])
                ax[nbrRow, nbrCol].set_xlabel(graphProperties["labelx"])
            elif(rows==1):
                ax[nbrCol].plot(dataX, dataY, graphProperties["color"])
                ax[nbrCol].set_title(graphProperties["title"])
                ax[nbrCol].set_ylabel(graphProperties["labely"])
                ax[nbrCol].set_xlabel(graphProperties["labelx"])
            fig.tight_layout()
            if nbrCol == (columns-1):
                nbrCol = 0
                nbrRow += 1
            else:
                nbrCol += 1
            if nbrRow == rows and nbrCol == columns:
                break
            if(graphProperties["flipy"] == True):
                plt.ylim(max(dataY), min(dataY))
        plt.show()
    
    def graphing(self, graphNameList, nbrofGraphs=0):
        if nbrofGraphs <= 0:
            nbrofGraphs = len(graphNameList)
            if nbrofGraphs == 0:
                print("no graphs to draw")
                return
        
        returns = self.findPositionAndSize(nbrofGraphs) # from the list of graphs we calculate the nbr of rows and columns to use while drawing
        if returns[0] == 1:
            rowColumn = returns[1]
            self.subPlots(rowColumn[0], rowColumn[1], graphNameList, (rowColumn[0]*rowColumn[1]))
            return

        elif returns[0] == 0:
            rowColumn = returns[2]
            self.subPlots(rowColumn[0], rowColumn[1], graphNameList, (rowColumn[0]*rowColumn[1]))
            print(graphNameList, "\n",graphNameList[8:])
            self.graphing(graphNameList[8:], returns[1])
        
    
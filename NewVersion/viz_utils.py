def visualize_contact_network(G, numNodes):
  spring_3D = nx.spring_layout(G, dim=3, seed=18)
  x_nodes = [spring_3D[i][0] for i in range(numNodes)]# x-coordinates of nodes
  y_nodes = [spring_3D[i][1] for i in range(numNodes)]# y-coordinates
  z_nodes = [spring_3D[i][2] for i in range(numNodes)]# z-coordinates
  edge_list = G.edges()

  #we  need to create lists that contain the starting and ending coordinates of each edge.
  x_edges=[]
  y_edges=[]
  z_edges=[]

  #need to fill these with all of the coordiates
  for edge in edge_list:
      #format: [beginning,ending,None]
      x_coords = [spring_3D[edge[0]][0],spring_3D[edge[1]][0],None]
      x_edges += x_coords

      y_coords = [spring_3D[edge[0]][1],spring_3D[edge[1]][1],None]
      y_edges += y_coords

      z_coords = [spring_3D[edge[0]][2],spring_3D[edge[1]][2],None]
      z_edges += z_coords

  #create a trace for the edges
  trace_edges = go.Scatter3d(x=x_edges,
                          y=y_edges,
                          z=z_edges,
                          mode='lines',
                          line=dict(color='black', width=2),
                          hoverinfo='none')

    #create a trace for the nodes
  trace_nodes = go.Scatter3d(x=x_nodes,
                          y=y_nodes,
                          z=z_nodes,
                          mode='markers',
                          marker=dict(symbol='circle',
                                      size=5,
                                      color='red', #color the nodes according to their community
                                      line=dict(color='black', width=0.5)),
                          hoverinfo='none'
                          )



  #we need to set the axis for the plot
  axis = dict(showbackground=False,
              showline=False,
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title='')

  #also need to create the layout for our plot
  layout = go.Layout(title="Boston University Contact Network",
                  width=1000,
                  height=800,
                  showlegend=False,
                  scene=dict(xaxis=dict(axis),
                          yaxis=dict(axis),
                          zaxis=dict(axis),
                          ),
                  margin=dict(t=100),
                  hovermode='closest')
    #Include the traces we want to plot and create a figure
  data = [trace_edges, trace_nodes]
  fig = go.Figure(data=data, layout=layout)

  fig.update_layout(scene=dict(xaxis_showspikes=False,
                              yaxis_showspikes=False, zaxis_showspikes=False))

  fig.show()

import ThreeDimViewer

viewport = ThreeDimViewer.ThreeDimViewer()
viewport.vertices = [(0, 0, 1), (0, 0, 2), (0, 1, 2), (1, 1, 2)]
viewport.colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1)]
viewport.cameras = []
viewport.main()

def map_data(self, map_name):
    temp_name = map_name.split('\\')
    map_name = ''
    for it in temp_name:
            map_name += '/' + it 
    filePath = self.data_dir + 'Maps' + map_name
    print(filePath)

    with gfile.Open(filePath, "rb") as f:
    #with gfile.Open(os.path.join(self.data_dir, "Maps", map_name), "rb") as f:
    #with gfile.Open('F:/game/StarCraft II/Maps/DefeatRoaches.SC2Map', "rb") as f:
        return f.read()

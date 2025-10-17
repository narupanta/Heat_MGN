import gmsh
import sys
import os
# Initialize Gmsh
class Cube:
    def __init__(self, length, width, height):
        self.length = length
        self.width = width
        self.height = height

def create_msh_file(output_dir, name, cube: Cube, size_min, dist_min, y_offset, num_track):
    gmsh.initialize()
    gmsh.logger.start()
    gmsh.model.add(name)

    # Box dimensions (in mm)
    length, width, height = cube.length, cube.width, cube.height

    # Create the box at the origin
    gmsh.model.occ.addBox(0, 0, 0, length, width, height)
    gmsh.model.occ.synchronize()

    # Generate laser tracks
    tracks = []
    if num_track == 1:
        p1 = gmsh.model.occ.addPoint(0, 0, height)
        p2 = gmsh.model.occ.addPoint(length, width, height)
        track = gmsh.model.occ.addLine(p1, p2)
        tracks.append(track)
    else:
        for i in range(num_track):
            y_pos = (i + 0.5) * (width / num_track)
            p1 = gmsh.model.occ.addPoint(0, y_pos, height)
            p2 = gmsh.model.occ.addPoint(length, y_pos, height)
            track = gmsh.model.occ.addLine(p1, p2)
            tracks.append(track)

    gmsh.model.occ.synchronize()

    # Tag surfaces for boundary conditions
    surfaces = gmsh.model.getEntities(dim=2)
    for surface in surfaces:
        gmsh.model.addPhysicalGroup(2, [surface[1]], tag=surface[1])
        gmsh.model.setPhysicalName(2, surface[1], "geometry surface")

    # Tag the volume
    volumes = gmsh.model.getEntities(dim=3)
    volume_ids = [v[1] for v in volumes]
    gmsh.model.addPhysicalGroup(3, volume_ids)
    gmsh.model.setPhysicalName(3, volume_ids[0], "geometry volume")

    # Mesh refinement around laser tracks
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", tracks)

    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", size_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", size_min * 5)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", dist_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", dist_min * 5)

    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

    # Generate the 3D mesh
    gmsh.model.mesh.generate(3)
    geometry_name = f"{name}_{cube.length}_{str(cube.width)}_{str(cube.height)}_{str(size_min)}_{str(dist_min)}_{str(y_offset)}_{str(num_track)}"
    gmsh.write(os.path.join(output_dir, f"{geometry_name}.msh"))

    # Optional GUI
    if "-nopopup" not in sys.argv:
        gmsh.fltk.run()
    with open(f"{geometry_name}_overview.txt", "w") as f:
        f.write("\n".join(gmsh.logger.get()))
    gmsh.finalize()
    return os.path.join(output_dir, f"{geometry_name}.msh")
# Example usage
if __name__ == "__main__":
    create_msh_file("./geometry/cube", Cube(2e-3, 2e-3, 1e-3), 0.04e-3, 0.25e-3, 1)

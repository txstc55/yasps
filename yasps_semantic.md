# Semantics and Syntax for YASPS

## Attributes Crush Intro
We first go over the definition of attributes.

Attribute always corresponds to some object (e.g. a scene, a mesh, vertices of mesh, faces of mesh etc.). Hence, we can think of attributes as a way to access properties of objects. As such, attributes are always associated with some object, and their initialization is always done through the following way:

```python
attribute = object.addAttribute(name, dimension = [...], ...)
```

Attributes can be explicitly initialized with value:
```python
attribute = object.addAttribute(name, dimension = [...], value = [...])
```

Under the hood, attributes have the following field
- `name` : name of the attribute.
- `dimension` : dimension of the attribute. For example, if we are defining position for vertices, the dimension is 3 instead of number of vertices times 3.
- `correspondance` : object to which the addAttribute() method is called upon.
- `value` : value of the attribute, which is always a 1d array when accessed. The dimension of the value is determined by the property of object and the dimension of the attribute.
- `kernel`: the cuda kernel used to compute this attribute.
- `lifted_from`: for attributes that are computed from primitives, we will touch it later on.

Outside of using addObject, we can access the object by:

```python
attribute = object[attribute_name]
```

We can also perform arithmetic operations on the attributes:
```python
attribute3 = object[attribute_name1] + object[attribute_name2]
attribute4 = attribute3 * 2
```

Or perform indexing operations:
```python
attribute5 = attribute4[0:10].concatenate(attribute3[10:20])
```

However, the following rules are imposed on the combination of attributes:
- The dimensions of the attributes must match.
- The correspondance must be the same.

This attribute can then be assigned the object again by:
```python
object.addAttribute(name, attribute = attribute5)
```

Note that this operation is valid only when the object is what the attribute corresponds to.

When we want to access the value of the attribute, we can use the following syntax:
```python
value = attribute5.compute().value()
```

Here, if `compute()` is not called, the value of the attribute will not updated and old values will be returned instead.

## Scene Creation
We first go over the creation of a scene.

```python
scene = yasps.Scene()
```

A scene itself contains some `meshObject`, a scene can also have `attribute`. So under the hood, a scene has the following fields:
- `meshObjects` : dictionary from mesh object name to mesh object.
- `attributes` : dictionary from attribute name to attributes.

Attributes are added to the scene by:
```python
scene.addAttribute(name, dimension = [...])
```



## Mesh Object Creation
We then go over the creation of a mesh object.

```python
m1 = scene.addMesh(name)
```

And now `m1` is a mesh object.

Like scene, mesh object also has a dictionary of attributes. The mesh object also has a hierachy index of 1. This means that the mesh object is a child of the scene. And any object that is added to the mesh object can also access the attributes of the scene.

We now will show the hierachy graph of how attrbutes can be accessed, at this point, we have scene and mesh object.

```text
|-- scene
|  |-- mesh1
|  |-- mesh2
|  |-- mesh3
```

A mesh contains two dictionaries:
- `attributes` : dictionary from attribute name to attributes.
- `primitives` : dictionary from primitive name to primitives.

Accessing a mesh in the scene can be done by:
```python
m1 = scene.m1
```

## Primitive Creation
A primitive is the smallest units in a mesh. For example, a vertex is a primitive as it depends on no other primitives. A face can also be a primitive, but it does not have to depend on vertex. The correlation between how three vertices become a face is done through `connectivity`, which we will discuss later.

Primitives are added to the mesh object by:
```python
primitive = m1.addPrimitive(name, num_primitives)
```

For example, if the mesh has 10 vertices, we can add a vertex primitive by:
```python
vertex = m1.addPrimitive('vertex', 10)
```

And if a mesh has 8 faces, we can add a face primitive by:
```python
face = m1.addPrimitive('face', 8)
```
Primitives can be accessed by:
```python
vertex = m1.vertex
face = m1.face
```

Primitives can also have attributes. For example, if we want to add position to vertices, we can do:
```python
position = vertex.addAttribute('position', dimension = 3, data = [[...],...])
```

Here, `dimension = 3` meaans that each position for each vertex has 3 values. The data itself, if flattened, has `NUM_VERTICES * 3` values.

In the same fashion, if we define some attributes for each face, we can do:
```python
random_attributes = face.addAttribute('random_attributes', dimension = 3, data = [[...],...])
```

## Connectivity
Connectivity is the way to define how primitives are connected to each other. For example, if we have a face, we can define how the face is connected to vertices. This is done by:
```python
m1.face.addConnectivity(to = m1.vertex, data = [[...], ...], dimension = 3)
```

Here, `to` is the primitive that the face is connected to. The data itself is a list of list. In the example, since we had 8 faces, the data is a list of 8 lists. Each list contains the indices of the vertices that the face is connected to. The dimension is 3, which means that each face is connected to 3 vertices.

We will also reserve `dimension = -1` for the case where the connectivity is not a fixed `m x n` matrix. For example, a vertex can connect to any number of faces. In this case, the data array's inner size is not fixed.

Connectivities do not explicitly have attributes. When a connectivity is added to a primitive, the primitive will now inherit the attributes of the primitive that it is connected to. For example, if we have a face that is connected to vertices, the face will now have the attributes of the vertices.


## Attribute Operations
Let us write a simple example that computes the normal of vertices by averaging the normals of faces that the vertex is connected to.

We first defint the scene, the mesh and the primitives.
```python
scene = yasps.Scene()
m1 = scene.addMesh('m1')
m1.addPrimitive('vertex', 4)
m1.addPrimitive('face', 4)
```

We now add the connectivity from face to vertex to construct a single tetrahedron.
```python
m1.face.addConnectivity(to = m1.vertex, data = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [2, 1, 3]], dimension = 3)
```

And we add the position attributes binded to the vertices.
```python
m1.vertex.addAttribute('position', dimension = 3, data = [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
```

We now access the position attribute for each triangle:
```python
position = m1.face["position"]
print(position.compute().value())
```

Here, since the face is connected to the vertices, the face will have the position attribute of the vertices. the operation
```python
m1.face["position"]
```
will now first check if `position` is inside `m1.face`'s attribute dictionary. As we never explicitly defined it, it will now check the connectivity list.

As we go through the connectivity list, we see that the face is connected to the vertices. So we now check if `position` is inside `m1.vertex`'s attribute dictionary. As we defined it earlier, we now know that `position` can be an attribute that is lifted from `m1.vertex`.

Now, we add the attribute named `position` to `m1.face`, mark that this attribute has a dimension of `3 by 3` since each face is connected to 3 vertices, and each position has 3 values. We also mark that the `lifted_from` attributes is `m1.vertex`, which means that the attribute is scattered from `m1.vertex`.

When we call `compute()`, a kernel to fetch those values will be generated and called upon, consequently updating the value of this attribute. The final output will be an array of size `4 x 3 x 3`, where each row represents the concatenated vertex positions of each triangle.

We now use the positions of each triangle to compute the normal of each triangle.
```python
def compute_normal(position):
    # position is a 3 by 3 representation of
    # the concatenated vertex positions of each triangle.
    # We first compute the normal of each triangle.
    v1 = position[:, 1] - position[:, 0]
    v2 = position[:, 2] - position[:, 0]
    normal = v1.cross(v2)
    normal = normal / normal.norm()
    return normal

m1.face.addAttribute('normal', attribute = compute_normal(m1.face['position']))
```

Here, we define a function `compute_normal` that takes in the position of each triangle and computes the normal of each triangle. We then add the attribute `normal` to the face primitive. The normal attribute has dimension `3` since each normal has 3 values. The value of the normal attribute is computed by calling the `m1.face["normal"].compute().value()` which will return a array of size `4 x 3` which represents the normal of each triangle.

Now we have the normal of each triangle, we will need to compute the normal of each vertex.
We first define a connectivity from vertex to triangle.
```python
m1.vertex.addConnectivity(to = m1.face, data = [[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]], dimension = -1)
```

Although in the tetrahedron, we know that each vertex corresponds to 3 triangles, we will still mark dimension as `-1` since the number of triangles that a vertex is connected to is not fixed in a general case.

Now we can access the normal for each triangle around a vertex by:
```python
normal = m1.vertex["normal"]
```
However, unlike `position` for triangle, now `normal` does not have a fixed dimension. This is because the number of triangles that a vertex is connected to is not fixed. As such, the operations that can be done for such attributes are limited to operations that can be done on a list of arrays of different sizes. For example:

```python
average_normal = m1.vertex["normal"].average().compute().value()
max_normal = m1.vertex["normal"].compute().value()
```
However, note that since the correspondance is still `m1.vertex`, those two operations are operations that are performed on each vertex instead of the entire mesh.

Another way to do this is to use a scatter operation:
```python
m1.vertex.addAttribute('vertex_normal_sum', dimension = 3)
m1.triangle['normal'].scatter(to = m1.vertex, attribute = 'vertex_normal_sum')
m1.vertex.addAttribute('num_triangles_around_vertex', dimension = 1, data = [3, 3, 3, 3])
m1.vertex.addAttribte('vertex_normal', attribute = m1.vertex['vertex_normal_sum'] / m1.vertex['num_triangles_around_vertex'])
```
When the `scatter` operation is performed, we will first check if the correspondance exists. Then the dimension check is performed. Since `vertex_normal_sum` is dimension `3`, and the `normal` of triangle is dimension `3 x 3`, and the correspondance is 1 triangle to 3 vertices, the dimension check will pass, and the operation is valid.

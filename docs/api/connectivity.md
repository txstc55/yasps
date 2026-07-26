# Connectivity API

Create connectivity through its source primitive:

```python
connectivity = source.addConnectivity(
  name,
  to=target,
  data=indices,
  dimension=arity,
)
```

## Fixed arity

For `dimension > 0`, `data` contains one tuple per source instance. The public
`value` is a flattened device array of the active prefix.

## Variable arity

For `dimension == 0`, `data` is a list of lists. YASPS stores:

- flattened `value` indices; and
- `compressedRows`, the CSR row offsets.

Use only with `SUM` or `AVERAGE` attribute construction.

## Properties

| Property | Description |
| --- | --- |
| `name`, `fullName` | Identity |
| `fromPrimitive` | Owner/source primitive |
| `toPrimitive` | Target primitive |
| `dimension` | Fixed arity or `0` for CSR |
| `value` | Flattened active indices |
| `compressedRows` | CSR offsets |
| `mesh`, `scene` | Hierarchy |

Generated index/CSR names are internal kernel interfaces.

## Updating

```python
connectivity.updateConnectivity(new_indices)
```

For dynamic fixed-arity topology:

```python
source.updateNumInstances(new_indices.shape[0])
if new_indices.shape[0]:
  connectivity.updateConnectivity(new_indices)
```

The implementation reuses a larger allocation when the new active topology
is smaller.

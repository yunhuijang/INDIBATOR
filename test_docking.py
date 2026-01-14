from tdc import Oracle

oracle = Oracle(name='gsk3b')

x = oracle('CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1')
assert abs(x - 0.03) < 0.0001

oracle = Oracle(name='JNK3')

x = oracle('C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O')
assert abs(x - 0.01) < 0.0001
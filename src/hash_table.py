'''
Hash table class and relevant parameter functions
'''


def get_id(item):
    '''Retrieves the ID, whether the input is an ID or block'''
    if type(item) == dict:
        return int(item['id'])
    else:
        return int(item)


def hash_node_and_ways(self, item):
    return hash(self.get_fn(item)) % len(self.table)


class HashTable:
    def __init__(self, len=10000000, hash_fn=None, get_fn=None):
        '''Set the hash function and get function (gets ID from the item)'''
        self.table = [[] for _ in range(len)]
        self.num_items = 0
        self.collisions = 0

        if hash_fn is not None:
            self.hash = hash_fn
        else:
            self.hash = hash_node_and_ways

        if get_fn is not None:
            self.get_fn = get_fn
        else:
            self.get_fn = get_id

    def insert(self, item):
        '''Hashes and inserts item, returning -1 if it's a duplicate'''
        index = self.hash(self, item)
        for i in self.table[index]:
            if item == i:
                return -1

        self.table[index].append(item)
        self.num_items += 1
        if len(self.table[index]) > 1:
            self.collisions += 1

    def get(self, id):
        '''Retrieves the node and ways item given the node ID'''
        index = self.hash(self, id)
        for item in self.table[index]:
            if self.get_fn(item) == id:
                return item
        return None

    def contains(self, id):
        '''Check if the list contains the given node ID'''
        index = self.hash(self, id)
        for item in self.table[index]:
            if self.get_fn(item) == id:
                return True
        return False

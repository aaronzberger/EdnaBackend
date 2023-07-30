'''Associate PA voter file addresses with PA address points file addresses:

A universe file is provided, for which addresses are from the PA voter export.

These addresses must be matched with addresses from the PA address points file
(which were previously associated with blocks in make_blocks.py).
'''

from typing import Optional
from src.config import houses_file_t, houses_file

import json


class Associater:
    houses_to_id: houses_file_t = json.load(open(houses_file))

    def __init__(self):
        self.failed_houses = self.apartment_houses = 0

    def associate(self, address: str) -> Optional[str]:
        '''
        Associate an address with a block ID

        Parameters:
            address (str): the address to associate

        Returns:
            Optional[str]: the block ID, or None if the address could not be associated
        '''
        if address not in self.houses_to_id:
            self.failed_houses += 1
            return None
        return self.houses_to_id[address]

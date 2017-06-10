#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import facebook

access_token = os.environ['ACCESS_TOKEN']

def main():
    graph = facebook.GraphAPI(access_token=access_token)
    profile_picture = graph.get_object(id='me/picture')

if __name__ == '__main__':
    main()
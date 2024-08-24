# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:16:36 2016

@author: kncrabtree
"""


class SpcatTransition:
    """
    A convenience class for representing SPCAT transitions
    
    The class constructor takes a string from a .cat file and parses it
    to extract all of the information associated with the transition.
    
    
    """
    def __init__(self,string):
        self.string = string
        w = [13,8,8,2,10,3,7,4,2,2,2,2,2,2,2,2,2,2,2,2]
        self.columns = []
        
        l = 0
        for i in range(len(w)):
            s = string[l:l+w[i]]
#            print(s)
            self.columns.append(s)
            l+=w[i]
            
        self.freq = float(self.columns[0])
        self.unc = float(self.columns[1])
        self.lgint = float(self.columns[2])
        self.int = 10.0**self.lgint
        self.dof = int(self.columns[3])
        self.egy = float(self.columns[4])
        self.gup = int(self.columns[5])
        self.tag = int(self.columns[6])
        self.fmt = int(self.columns[7])
        self.uqns = [ None if self.columns[8+i]=='  ' else int(self.columns[8+i]) for i in range(6) ]
        self.lqns = [ None if self.columns[14+i]=='  ' else int(self.columns[14+i]) for i in range(6) ]
        
        
def parse_cat(filename):
    transitions = []
    with open(filename,'r') as catfile:
        for line in catfile:
            t = SpcatTransition(line)
            t.string = line
            transitions.append(t)
            
    return transitions
        
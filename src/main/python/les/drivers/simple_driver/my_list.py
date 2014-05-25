class Pair:
	 def __init__(self, key = None, weight = None):
	 	self.key = key
	 	self.weight = weight

class ListElement:
  def __init__(self, value = None, prev = None, next = None):
    self.value = value
    self.prev = prev
    self.next = next

class MyList:
  def __init__(self):
    self.beg = None
    self.end = None
    self.size = 0
    
  def Add(self, x):
    if self.size == 0:
      self.beg = self.end = ListElement(x, None, None)
    else:
      new_el = ListElement(x, self.end, None)
      self.end.next = self.end = new_el #
    self.size += 1
    
  def Del(self, x):
    if self.size == 0:
    	print "ERROR while deleting!" 
    	return
    if self.size == 1:
    	if self.beg != x:
    		print "ERROR: There is no such x!"
    		return 
    	self.__init__()
    elif x == self.beg:
      self.beg = self.beg.next
    elif x == self.end:
      self.end = self.end.prev
    else:
      x.prev.next = x.next
      x.next.prev = x.prev
      del(x)
    self.size -= 1
    
  def FindMax(self):
  	heaviest = self.beg
  	tmp = self.beg.next
  	while tmp != None:
  		if heaviest.value.weight < tmp.value.weight:
  			heaviest = tmp # copy
  		tmp = tmp.next
  	return heaviest

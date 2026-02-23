from collections import deque
s=open('app.py','r',encoding='utf-8').read()
pairs={'(':')','[':']','{':'}'}
openers=set(pairs.keys())
closers=set(pairs.values())
stack=[]
for i,ch in enumerate(s,1):
    if ch in openers:
        stack.append((ch,i))
    elif ch in closers:
        if not stack:
            print('Unmatched closer',ch,'at',i)
            break
        last,li=stack.pop()
        if pairs[last]!=ch:
            print('MISMATCH: opener',last,'at',li,'vs closer',ch,'at',i)
            break
else:
    if stack:
        print('Unclosed opener',stack[-1])
    else:
        print('All balanced')

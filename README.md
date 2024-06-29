# icfpc2024

My code for ICFPC contest https://icfpcontest2024.github.io. Not sure if I'll spend more time on it, current changes wrap up the ideas I had

## ICFP expressions and their usage

I've implemented almost everything, except `B\$` and lambda evaluation.

Main code  for ICFP expressions is in the `src` directory. The code which uses them, parses them and tries to communicate with the server is in the `notebooks` directory (extremly messy, subject to changes, might be not reproducible). 

## Lambdaman

I've solved first several tasks manually, for now the most promising idea is to use minimal spanning tree to produce optimal route

Update: approach solves some tasks, however, doesn't work well due to recursion depth and doesn't work well on tasks where the data was not given in string format (I haven't implemented this part so let's skip it for now).

Ideas to implement

- [x] It can be noted that when we visit last branch of the spanning tree, we might want to stop when we've seen all the vertices.
- [ ] Also, if the last branch has the largest size, we don't do more moves when the last branch is not very large. So it is useful to sort branches by their size before starting to traverse. Of course this helps only if we don't return to the position where we started



## Spaceship

It seems obvious that in this task we just have to sort the desired positions in the right way, however, I haven't thought about how to deal with the situation when some positions near each other, or situations when some positions are missing. I don't want to solve each of the puzzles as a separate complex optimization task.

I've also solved first several tasks manually.
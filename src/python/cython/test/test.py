import bla
import Timer

n = 1000

with Timer.Timer('intense ...', digits=3):
    bla.intense(10000)

with Timer.Timer('not_intense ...', digits=3):
    bla.not_intense(10000)


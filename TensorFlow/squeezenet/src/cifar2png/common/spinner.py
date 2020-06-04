#!/usr/bin/env python
# coding: utf-8

import sys
import time
import threading

class Spinner:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1:
            for cursor in '|/-\\': yield cursor

    def __init__(self, delay=None, prefix=None):
        self.prefix = prefix
        self.prefix_len = len(prefix)
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay): self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(self.prefix)
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\r')
            sys.stdout.flush()

    def start(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def stop(self):
        self.busy = False
        sys.stdout.write('\r')
        sys.stdout.write(' ' * (self.prefix_len + 1))
        sys.stdout.write('\r')
        sys.stdout.flush()
        time.sleep(self.delay)

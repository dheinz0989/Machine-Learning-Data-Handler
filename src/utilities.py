'''

This Module provides some Function which are considered util. They are not necessary to run the code but ease it a lot.

In total the following different Utilities classes are provided:
 - Logger: this class initializes a logging instance which can be used to log all activities.
 - Decorators: this class provides a set of different Decorators which can be used to add functionalities to functions
 - ClassAttrHandler: this class provides some functionalities with respect to classes and their respective attributes.
 - Dict_to_Obj: converts a dictionary to an object notation

'''
import logging
import sys
import time
from functools import wraps
import yaml
from pathlib import Path

__all__ = [
    'Logger',
    'Decorators',
    'YamlParser'
]


class Logger:
    """
    This class adds a logging instance which can be imported in other modules and used to track code and activities.
    It consists of a single function and is only embedded in a class for giving a namespace that clarifies that is is a logging instance.
    All logs to are written to stdout. Furthermore, logs can optionally be written to a logging file
    The logging file is identified via a timestamp and written into ./logs/

    Usage: Import this class at the beginning of a module. You can then access the log attribute and use it as a logging instance
    Example::

    > 1 from Utilities import Logger
    > 2 log = Logger.log()
    > 3 log.info('Control is here')
    > # log prints "Control is here"
    """

    @staticmethod
    def initialize_log(write_to_file=False):
        """
        Initializes a logging instance that writes to stout. It can optionally also write to a logging file

        :param write_to_file: indicates, if a subdirectory with "logs" is  created in which a logging file is written into
        :type write_to_file: bool
        :return:
        """
        if write_to_file:
            # Create a logging directory and a log file name
            Path('logs').mkdir(parents=True, exist_ok=True)
            log_file_name = f'logs/log_{__name__}_{time.strftime("%Y-%m-%d", time.gmtime())}.log'
            handlers = [
                logging.FileHandler(log_file_name),
                logging.StreamHandler(sys.stdout),
            ]
        else:
            handlers = [logging.StreamHandler(sys.stdout)]

        logging.basicConfig(level=logging.INFO,
                                format="%(asctime)s,%(msecs)d - file: %(module)s  - func: %(funcName)s - line: %(lineno)d - %(levelname)s - msg: %(message)s",
                                datefmt="%H:%M:%S",
                                handlers=handlers,
                                )
        return logging.getLogger(__name__)


log = Logger.initialize_log()


class Decorators:
    """
    This class provides a set of functionality with respect to decorate functions. These decorators are considered
    util as they prevent to repeat the same code, add functionality to a function on the fly, allows a lot of type
    and input checking and so on.

    All the functions defined inside this class take a function as an input and return a decorated function.
    """

    @staticmethod
    def run_time(func):
        """
        When decorating a function with this decorator, it indicates the function's run time in a hh:mm:ss after
        the function returns.

        Example::

        > # Assume the function needs exactly 1 minute, 13.534 seconds to execute
        > @Decorators.run_time
        > 1 def foo(x):
        > 2   ...
        > ...
        > 7 foo(10)
        > #console prints "00:01:13,534"

        :param func: function to decorate
        :return: decorated function which indicates function run time
        """
        assert callable(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wraps the original function and adds the decorator's run time display functionality
            """
            start = time.time()
            ret = func(*args, **kwargs)
            end = time.time()
            m, s = divmod(end - start, 60)
            h, m = divmod(m, 60)
            ms = int(s % 1 * 1000)
            s, m, h = int(round(s, 0)), int(round(m, 0)), int(round(h, 0))
            log.info(
                f'Execution Time (hh:mm:sec) for function "{func.__name__}": {h:02d}:{m:02d}:{s:02d},{ms:03d}'
            )
            return ret

        return wrapper

    @staticmethod
    def show_args(func):
        '''
        When decorating a function with this decorator, it indicates the arguments passed to the function.

        Example::

        > @Decorators.show_args
        > 1 def foo(x):
        >       ....
        >    10 foo(10)
        >    11 #console prints "Executing 'foo' with args 10 and ''"

        :param func: function to decorate
        :return: decorated function which indicates function's arguments
        '''
        assert callable(func)
        @wraps(func)
        def wrapper(*args, **kwargs):
            log.info(f"Executing '{func.__name__}' with args {args} and {kwargs}")
            ret = func(*args, **kwargs)
            return ret
        return wrapper

    @staticmethod
    def accepted_arguments_within_class_methods(accepted_args):
        '''
        When decorating a function with this decorator, the function's arguments are checked against a list of valid arguments.
        If an invalid argument is encountered, the function is not executed. This decorator is basically the same as "accepted_arguments"
        decorator except that it is aimed for functions within classes (i.e. containing a "self" parameter). In these setup, the class instance
        itself is passed as the first argument. Therefore, this Decorator only checks the second till last argument for correctness.

        Example::

        > 1 class Foo:
        > 2   ...
        > ...
        > 10   @Decorators.accepted_arguments_within_class_methods([0,1])
        > 11   def bar(self):
        > 12         ...
        > ...
        > 18 foo=Foo(1)
        > 19 foo.bar()
        > # console prints: Encountered a non-valid argument.
        > # console prints: Valid arguments are: [0,1]

        :param accepted_args: list of accepted arguments by the function
        :type accepted_args: list
        :return: a decorated function which checks the aguments
        '''

        def decorator(func):
            @wraps(func)
            def wrapper(*args):
                try:
                    assert all([a in accepted_args for a in args[1:]])
                except AssertionError:
                    raise SyntaxError(f'Encountered a non-valid argument.\nValid arguments are: {accepted_args}')
                result = func(*args)
                return result

            return wrapper

        return decorator

    @staticmethod
    def counter(func):
        '''
        When decorating a function with this decorator, it indicates how often the function has been called.

        Example::

         >   @Decorators.counter
         >   1 def foo(x):
         >       ....
         >   10 foo(10)
         >   11 #console prints "Executing 'foo' with args 10 and ''"

        :param func: function to decorate
        :return: decorated function which indicates how often the function has been called
        '''
        assert callable(func)
        @wraps(func)
        def wrapper(*args, **kwargs):
            wrapper.count = wrapper.count + 1
            res = func(*args, **kwargs)
            log.info(f"Number of times '{func.__name__}' has been called: {wrapper.count}x")
            return res
        wrapper.count = 0
        return wrapper

    @staticmethod
    def argument_in_dictionary_key(dictionary: dict):
        '''
        When decorating a function with this decorator, the function's arguments are checked against a list of valid arguments.
        If an invalid argument is encoutered, the function is not executed.

        Example::

        > @Decorators.accepted_arguments([0,1])
        > 1 def foo(x):
        >    ...
        > 7 foo(10)
        > # console prints: Encountered a non-valid argument.
        > # console prints: Valid arguments are: [0,1]

        :param accepted_args: list of accepted arguments by the function
        :type accepted_args: list
        :return: a decorated function which checks the aguments
        '''

        def decorator(func):
            @wraps(func)
            def wrapper(*args):
                accepted_keys = list(dictionary.keys())
                try:
                    assert all([a in accepted_keys for a in args])
                except AssertionError:
                    errounos_args = [a for a in args if a not in accepted_keys]
                    raise KeyError(
                        f'Encountered the non-valid argument "{errounos_args[0]}" which is not a dictionary key. Valid dictionary keys arguments are: {accepted_keys}') if len(
                        errounos_args) == 1 else KeyError(f'Encountered the following non valid arguments {[err for err in errounos_args]} which are not dictionary keys. Valid dictionary keys arguments are: {accepted_keys}')
                result = func(*args)
                return result

            return wrapper

        return decorator


class YamlParser:
    def __init__(self, yaml_file):
        self.yaml_file = yaml_file
        self.file_content = None

    def read(self):
        with open(self.yaml_file, 'r') as stream:
            self.file_content = yaml.safe_load(stream)
        return self

    def get_classes_by_key(self, key):
        return self.convert_nested_dictionary_class(self.file_content[key])

    def get_file(self):
        return self.file_content

    @staticmethod
    def str_to_class(classname):
        return getattr(sys.modules[__name__], classname)

    @staticmethod
    def convert_nested_dictionary_class(d):
        for k, v in d.items():
            if isinstance(v, dict):
                YamlParser.convert_nested_dictionary_class(v)
            else:
                d[k] = YamlParser.str_to_class(v)
        return d


def inspect_obj(obj):
    # TODO: option to hide __
    log.info(f"Object {obj} has the following attributes and values")
    for attr in dir(obj):
        try:
            print(f"obj.{attr} = {getattr(obj,attr)}")
        except Exception as e:
            log.error(e)
            raise e
### Python基础
1. #### 异常处理
    ```py
    try:
        检测范围
    except Exception[as reason]:
        出现异常处理代码
    final:
        无论如何都会执行的代码
    ```
2. #### __类__
    - 特点：<br>
        python方法需实例化才能被调用
    - 成员：
        - [ ] 私有变量<br>
            变量或函数前加__就变成私有，但仍可通过 ```_类名__变量/函数名```调用
            ```py
            class Person:
                __name = "张三"
                def getName(self):
                    return self.__name
                    
            # 调用
            P = Person()
            P.__name # 错误，不可访问
            P._Person__name  # 可以访问
            
            ```
        - [ ] 保护变量<br>
            变量或函数前加_，只有类和子类可以访问，import不可导入
            ```py
            class Person:
                _name = "张三"
                def getName(self):
                    return self.__name
                    
            # 调用
            P = Person()
            P._name # 可访问
            
            ```
        - [ ] 类对象<br>
            在__init__()外，相当于c++的成员变量，在类对象实例化后会覆盖类对象成为新的成员
            ```py
            class Person:
                age = 0
                def addAge(self):
                    self.age += 1
                    
                    
            # 调用
            p1 = Person()
            p2 = Person()
            p2.addAge()
            Person.age = 10
            p1.age # 结果为10
            p2.age # 因为已被实例化，值为1 
            ```
        - [ ] 初始化类
            ```py
            class Person:
                def __init__(self,) # 相当于c++的构造函数
            ```
        - [ ] 构造类<br>
            ```__new__(cls,[,...])``` 在```__init__```前调用创建类实例，python3默认继承object
            ```py
            class CapStr(str):
                def __new__(cls,string):
                    string = string.upper()
                    return str.__new__(cls,string)
            
            ```
        - [ ] 析构<br>
            ```__del__(self)``` 垃圾销毁时自动调用
        - [ ] *注意*:属性与方法名字相同，属性会覆盖方法
    - 重载算术运算
        - [ ] 正常算术运算
            |方法 |功能 |
            |:-:|:-:|
            | \_\_add__(self,other)|加法 |
            |\_\_sub__(self,other)|减法 |
            |\_\_mul__(self,other)|乘法 |
            |\_\_truediv__(self,other)|/ |
            | \_\_floordiv__(self,other)|// |
            | \_\_mod__(self,other)|% |
            | \_\_divmod__(self,other)|divmod |
            | \_\_pow__(self,other[,modulo])|** |
            | \_\_lshift__(self,othrt)| 左移<<|
            |\_\_rshift__(self,other) | 右移>>|
            |\_\_and__(self,other) | &|
            | \_\_xor__(self,other)| 异或^|
            | \_\_or__(self,other)|\||
        - [ ] 反运算<br>
            当做操作数不支持时调用有操作数的方法
            |方法 |功能 |
            |:-:|:-:|
            | \_\_radd__(self,other)|右操作数加法 |
        
        - [ ] 增值操作<br>
            自增方法
            |方法 |功能 |
            |:-:|:-:|
            | \_\_iadd__(self,other)|+= |
        - [ ] 一元操作
            |方法 |功能 |
            |:-:|:-:|
            | \_\_neg__(self)|定义负号 |
            | \_\_pos__(self)|定义正号|
            | \_\_abs__(self)|绝对值 |
            | \_\_invert__(self)|按位取反 |
        - [ ] 类型转换
            |方法 |功能 |
            |:-:|:-:|
            | \_\_int__(self)|强制转换成整数 |
            | \_\_float__(self)|强制转换小数|
            | \_\_round__(self,[,n])|四舍五入 |
    - 魔法函数
        - [ ] 字符函数
            ```py
            __str__(self) # 调用print时回调用
            
            __repr__(self) # 调用情况如下
            # Nint为自定义类
            P = Nint(5)
            P # 调用__repr__(self)
            ```
        - [ ] 属性访问
            ```py
            __getattr__(self,name)
            # 定义当用户试图获取一个不存在属性时的行为
            
            __getattribute__(self,name)
            # 定义当该类属性被访问时行为
            
            __setattr__(self,name,value)
            # 定义当一个属性被设置时行为
            
            __delattr__(self,name)
            # 定义一个属性被删除时行为
            ```
        - [ ] 描述符<br>
            将某种特殊类型类实例指派给另一个类的属性<br>
            特殊类型要具有以下方法
            ```py
            __get__(self,instance,ower)
            # 用于访问属性，返回属性值
            
            __set__(self,instance,value)
            # 在属性分配操作中调用，不返回任何内容
            
            __delete__(self,instance)
            # 控制删除操作，不返回任何内容
            ```
            通过以上方法可实现通过属性设置属性
            ```py
            property(fget = None. fset = None, fdel = None)
            # fget->获取属性 fset->设置属性 fel->删除属性
            
            # 举个例子
            class C:
                def __init__(self,size = 10):
                    self.size = size
                    
                def getSize(self):
                    return self.size
                
                def setSize(self,val):
                    self.size = val
                
                def delSize(self):
                    del self.size
                
                x = property(getSize,setSize,delSize)
                
            # 调用
            c1 = C()
            c1.x # 相当于c1.getSize()
            c1.x = 18 # 相当于c1.setSize(18)
            del c1.x # 相当于c1.delSize()
                
            ```
        - [ ] 定制序列
            - 若定制容器不可变
                ```python
                # 需要定义函数
                __len__(self) # 相当于len()
                
                __getitem__(self,key) # 相当于self[key]
                
                ```
            - 若定制容器可变
                ```python
                # 除上述__len__ __getitem__ 还需
                __setitem__(self,key,value) # 相当于self[key] = value
                
                __delitem(self,key) # 相当于del self[key]
                ```
        - [ ] 迭代器
            ```python
            iter() # 返回迭代器
            __iter__(self)
            
            next() # 迭代器规则
            __next__(self)
            
            # 举个例子
            string = "hello"
            it = iter(string)
            next(it) # 返回值，若到头还调用则抛出异常
            ```
    - 一些函数
        ```python
        issubclass(class,classinfo)
        # classinfo为元组，class是其中任意一个候选类的子类就返回true
        
        isinstance(object,classinfo)
        # classinfo可为元组，查看实例对象是否属于某类，若为元组，符合其中一个就可以
        
        hasattr(object,name)
        # 测试object类是否具有name属性，name应为字符串
        
        getattr(object,name[,default])
        # 获取属性值，若没有则返回默认值
        
        setattr(object,name,value)
        # 设定属性值
        
        delattr(object,name)
        # 删除属性
        ```
    - 继承
        - [ ] 语法
            ```py
            # DerivedClassName子类 BaseClassName所要继承的父类
            class DerivedClassName(BaseClassName):
                ...
                
            # 多重继承
            class DerivedClassName(Base1,Base2,...):
                ...
            
            ```
        - [ ] 特点 
            - 子类中定义与父类同名的方法或属性会自动覆盖父类对应方法或属性
        - [ ] 保持父类某些功能并添加新功能
            ```py
            class Fish:
                def __init__(self):
                    self.x = 10
                    self.y = 20
            ```
            - 调用未绑定父类方法
                ```py
                class Shark(Fish):
                    def __init__(self):
                        Fish.__init__(self)
                        self.hungry = True
                ```
            - 使用super<br>
                不止初始函数，还可以super其他成员函数
                ```py
                class Shark(Fish):
                    def __init__(self):
                        super().__init__()
                        self.hungry = True
                ```
3. #### fun(*arg,**kwargs)<br>
   *args:把参数格式化存储在一个元组中，长度没有限制，必须位于普通参数和默认参数之后<br>
**kwargs:参数格式化存储在一个字典中，必须位于参数列表最后面
4. #### 元组和列表<br>
    元组不可变，列表可变，元组可被哈希
5. #### 深浅拷贝<br>
    浅拷贝：在另一地址创建新的变量或容器，但容器内元素均是源对象元素地址的拷贝<br>
深拷贝：容器内元素地址也是新开辟的 
6. #### 内存管理
- 对象引用机制：<br>
    python内部使用引用计数，来保持追踪内存中的对象，所有对象都有引用计数
- 引用计数：<br>
  - 引用计数增加：<br>
     一个对象分配一个新的名称<br>
将其放入容器中
  - 引用计数减少<br>
     使用del语句对对象别名显示销毁<br>
引用超出作用域或重新赋值
- 垃圾回收<br>
  当一个对象引用计数归0，它将被垃圾收集机制处理掉<br>
具有对象循环引用或全局命名空间引用的变量，在python退出往往不被释放
7. #### 闭包<br> 
    在一个内部函数中，对外部函数作用域变量进行引用
8. #### 函数装饰器<br>
    可以让其他函数在不需要做任何代码改动前提下，增加额外功能，装饰器返回值也是一个函数对象，常用于插入日志，事务处理。
    ```python
    '''
    举例说明
    '''
    import logging
    def use_logging(func):
        def wrapper():
            logging.warn('%s is running'%fun.__name__)
            return func()
        return wrapper
    @use_logging  #语法糖
    def fun()
        print('I am foo')
    foo()
    ```
9. #### 生成器迭代器
- 迭代器：不把所有元素装载到内存中，等到调用next才返回该元素
- 生成器：本质还是一个迭代器，yeild对应值被调用不会立刻返回，而是调用next方法时再返回
- 举例说明：range和items()返回都是列表，xrange(),iteritems()返回是迭代器
  
10. #### 匿名函数<br>
    无名，用完就完
11. #### 回调函数通信<br>
    把函数指针作为参数传递给另一个函数，将整个函数当作一个对象赋值给调用函数
12. #### python2,3区别：
- print在3中变为函数，2中是语句
- 编码不同，2是asscii，3是utf8
- xrange,range不同

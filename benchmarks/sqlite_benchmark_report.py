import time
import random
import inspect
import sqlite3
import matplotlib.pyplot as plt


import qcodes as qc


from profiling_sqlite_base import benchmark_add_results_vs_MAX_VARIABLE_NUMBER


MAX_VARIABLE_NUMBER = 250000 - 1
DB_PATH = 'executemany_benchmarks.db'

def create_table_manyrows(cols=1000):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        r = ['val_%s INTEGER' % i for i in range(cols)]
        sql = f'CREATE TABLE IF NOT EXISTS lol2 (id INTEGER PRIMARY KEY, {",".join(r)})'
        cur.execute(sql)


def drop_table_manyrows():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute('DROP TABLE IF EXISTS lol2')


def drop_and_create_table(cols=1000):
    drop_table_manyrows()
    create_table_manyrows(cols)


def bench_executemanyrows_execute_many(rows=2000000, cols=1000):
    drop_and_create_table(cols)
    data = [[i for j in range(cols)] for i in range(rows)]
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        t1 = time.time()
        sql = f'INSERT INTO lol2 VALUES (NULL, {",".join(["?" for i in range(cols)])})'
        cur.executemany(sql, data)
    return time.time() - t1


def bench_executemanyrows_execute(rows=2000000, mvn=MAX_VARIABLE_NUMBER, cols=4):
    drop_and_create_table(cols)
    values = [[j for i in range(cols)] for j in range(rows)]
    no_of_rows = rows
    no_of_columns = cols

    rows_per_transaction = int(int(mvn)/no_of_columns)

    _columns = "id, " +",".join(['val_%s' % i for i in range(cols)])
    _values = "(NULL, " + ", ".join(["?"] * len(values[0])) + ")"

    a, b = divmod(no_of_rows, rows_per_transaction)
    chunks = a*[rows_per_transaction] + [b]
    if chunks[-1] == 0:
        chunks.pop()

    start = 0
    stop = 0
    qurs, vals = [], []
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        for ii, chunk in enumerate(chunks):
            _values_x_params = ",".join([_values] * chunk)

            query = f"""INSERT INTO lol2
            ({_columns})
            VALUES
            {_values_x_params}
            """
            stop += chunk
            flattened_values = [item for sublist in values[start:stop]
                                for item in sublist]
            qurs.append(query)
            vals.append(flattened_values)
            start += chunk
        t1 = time.time()
        for q, v in zip(qurs, vals):
            cur.execute(q, v)
    return time.time() - t1


def bench_executemanyrows_executescript(rows=2000000, cols=4):
    drop_and_create_table(cols)
    data = [[i for j in range(cols)] for i in range(rows)]
    lel = [",".join(map(str, d)) for d in data]
    insert_stmts = "\n".join([f"INSERT INTO lol2 VALUES (NULL, %s);" % i for i in lel])
    sql = f"""BEGIN TRANSACTION;
{insert_stmts}
COMMIT TRANSACTION;
"""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        t1 = time.time()
        cur.executescript(sql)

    return time.time() - t1


def plot_insertion_speed_vs_nrows(cols, maxrows, stepsize, mvn1, mvn2):
    plt.figure()
    rows = range(1, maxrows, maxrows//stepsize)
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    for row in rows:
        t1.append(bench_executemanyrows_execute_many(rows=row, cols=cols))
        t2.append(bench_executemanyrows_execute(rows=row, cols=cols, mvn=mvn1))
        t3.append(bench_executemanyrows_executescript(rows=row, cols=cols))
        t4.append(bench_executemanyrows_execute(rows=row, cols=cols, mvn=mvn2))
    plt.plot(rows, t1, label='executemany')
    plt.plot(rows, t2, label='execute as in sqlite_base with mvn=%s' % mvn1)
    plt.plot(rows, t3, label='executescript')
    plt.plot(rows, t4, label='execute as in sqlite_base with mvn=%s' % mvn2)
    plt.legend()
    plt.xlabel('number of rows')
    plt.ylabel('insertion time $\mathit{S}$')
    fname = 'insertion_speed_vs_nrows_%s_%s.png' % (cols, maxrows)
    plt.savefig(fname)
    return fname


def plot_insertion_speed_vs_ncols(rows, maxcols, stepsize, mvn1, mvn2):
    plt.figure()
    cols = range(1, maxcols, maxcols//stepsize)
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    for col in cols:
        t1.append(bench_executemanyrows_execute_many(rows=rows, cols=col))
        t2.append(bench_executemanyrows_execute(rows=rows, cols=col, mvn=mvn1))
        t3.append(bench_executemanyrows_executescript(rows=rows, cols=col))
        t4.append(bench_executemanyrows_execute(rows=rows, cols=col, mvn=mvn2))
    plt.plot(cols, t1, label='executemany')
    plt.plot(cols, t2, label='execute as in sqlite_base with mvn=%s' % mvn1)
    plt.plot(cols, t3, label='executescript')
    plt.plot(cols, t4, label='execute as in sqlite_base with mvn=%s' % mvn2)
    plt.legend()
    plt.xlabel('number of columns')
    plt.ylabel('insertion time $\mathit{S}$')
    fname = 'insertion_speed_vs_ncols_%s_%s.png' % (rows, maxcols)
    plt.savefig(fname)
    return fname


def plot_insertion_speed_vs_max_var_num():
    plt.figure()
    fname = 'plot_insertion_speed_vs_max_var_num.png'
    t1 = []
    a = qc.SQLiteSettings.limits['MAX_VARIABLE_NUMBER']
    mvn = range(2, a, a//50)
    for i in mvn:
        t1.append(bench_executemanyrows_execute(rows=100000, mvn=i, cols=2))
    plt.plot(mvn, t1)
    plt.xlabel('MAX_VARIABLE_NUMBER')
    plt.ylabel('insertion time $\mathit{S}$')
    plt.savefig(fname)
    return fname

def insert_commit_1(n=200000):
    # commit for every insert
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    t1 = time.time()
    for i in range(n):
        cur.execute('INSERT INTO lol2 VALUES (NULL, ?)', (42,))
        conn.commit()
    t2 = time.time()
    conn.close()
    return t2 - t1


def insert_commit_2(n=200000):
    # commit once after each individual insertion
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    t1 = time.time()
    for i in range(n):
        cur.execute('INSERT INTO lol2 VALUES (NULL, ?)', (42,))
    conn.commit()
    t2 = time.time()
    conn.close()
    return t2 - t1


def insert_commit_3(n=200000):
    # executemany and commit afterwards
    r = (42,)
    res = [r for i in range(n)]
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    t1 = time.time()
    cur.executemany('INSERT INTO lol2 VALUES (NULL, ?)', res)
    conn.commit()
    t2 = time.time()
    conn.close()
    return t2 - t1


def benchmark_inserts():
    plt.figure()
    fname = 'benchmark_inserts.png'
    n = range(1, 1000, 100)
    res1 = []
    res2 = []
    res3 = []
    drop_and_create_table(1)
    for i in n:
        drop_and_create_table(1)
        res1.append(insert_commit_1(i))
        drop_and_create_table(1)
        res2.append(insert_commit_2(i))
        drop_and_create_table(1)
        res3.append(insert_commit_3(i))

    plt.plot(n, res1, label='insert and commit every time')
    plt.plot(n, res2, label='insert every time and commit once')
    plt.plot(n, res3, label='insertmany and commit once')
    plt.legend()
    plt.xlabel('inserted rows')
    plt.ylabel('insertion time $\mathit{S}$')
    plt.savefig(fname)
    return fname


def average_lookup(m=2000, n=200000):
    # average lookup
    insert_commit_3(n)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    indices = list(range(n))
    random.shuffle(indices)
    getind = indices[:m]
    t1 = time.time()
    for i in getind:
        cur.execute('SELECT * FROM lol2 WHERE id=?', (i,))
        cur.fetchall()
    t2 = time.time()
    return t2 - t1


def average_lookup_commit(m=2000, n=200000):
    # average lookup with commit after each select
    insert_commit_3(n)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    indices = list(range(n))
    random.shuffle(indices)
    getind = indices[:m]
    t1 = time.time()
    for i in getind:
        cur.execute('SELECT * FROM lol2 WHERE id=?', (i,))
        cur.fetchall()
        conn.commit()

    t2 = time.time()
    return t2 - t1


def benchmark_lookups():
    plt.figure()
    fname = 'benchmark_lookups.png'
    t1 = []
    t2 = []
    tot = 1000000
    drop_and_create_table(1)
    insert_commit_3(tot)
    n = range(1, tot, tot//10)
    m = [i//20 for i in n]
    for i, j in zip(n, m):
        t1.append(average_lookup(m=j, n=i))
        t2.append(average_lookup_commit(m=j, n=i))
    plt.plot(n, t1, label='select every time')
    plt.plot(n, t2, label='select and commit every time')
    plt.legend()
    plt.xlabel('number of records')
    plt.ylabel('lookup time $\mathit{S}$')
    plt.savefig(fname)
    return fname


def header(text):
    return '\n' + text + '\n' + '=' * len(text) + '\n\n'


def sheader(text):
    return '\n' + text + '\n' + '-' * len(text) + '\n\n'


def image(path):
    return '\n' + '.. image:: %s\n\n' % path


def text(tex):
    return tex + '\n'


def codeline(text):
    return '\n.. code:: python\n\n' + '\t' + text + '\n\n'

def codefunction(func):
    source = inspect.getsource(func)
    return '\n.. code:: python\n\n\t' + source.replace('\n','\n\t') + '\n\n'

def make_report():
    report = []
    report.append([header('Sqlite insertion benchmarks')])

    # benchmark committing
    benchmark_inserts_1 = benchmark_inserts()
    report.append([sheader('Benchmarking the effect of committing'),
                   image(benchmark_inserts_1),
                   text('The orange line lies below the green line'),
                   text('The blue and orange graphs each have'),
                   codeline('cur.execute(query, value)'),
                   text('inside a for loop, while the green graph uses'),
                   codeline('cur.executemany(query, values)'),
                   text('The difference between the blue and orange graph is that'
                        ' we commit for every insertion in the blue and we '
                        'commit only once for the orange after all insertions.'),
                   text('From this we see that it is way faster to commit only once '
                        'after all the insertions have been executed. The difference '
                        'between execute and executemany is minimal.')])

    benchmark_lookups_1 = benchmark_lookups()
    report.append([sheader('Benchmarking for random lookups'),
                   image(benchmark_lookups_1),
                   text('There is no overhead of comitting when doing lookups.'),
                   text('On the other hand there is no reason to commit when there are no changes.')])

    # benchmark insertion speed
    cols = 2
    maxrows = 400 * 600
    mvn2 = qc.SQLiteSettings.limits['MAX_VARIABLE_NUMBER']
    mvn1 = 999

    is_vs_nrows_1 = plot_insertion_speed_vs_nrows(cols, maxrows, 20, mvn1, mvn2)
    report.append([sheader('Insertion speed vs. number of rows'),
                   image(is_vs_nrows_1),
                   text('This plot shows the insertion speed for %s columns' % cols)])

    report.append([text('The lines correspond to the three methods'),
                   codeline('cur.executemany(query, values)'),
                   codeline('cur.execute(query, values)'),
                   codeline('cur.executescript(script)'),
                   text('The execute method call is used as in sqlite_base.'),
                   text('The variable mvn stands for the MAX_VARIABLE_NUMBER from the SQLiteSettings.'),
                   text('Wee see that the fastest approach is the one already implemented, but with a mvn of '
                        '%s. For mvn=%s this approach is actually the slowest. ' % (mvn1, mvn2))])
    cols = 100
    maxrows = 400 * 600
    is_vs_nrows_2 = plot_insertion_speed_vs_nrows(cols, maxrows, 20, mvn1, mvn2)
    report.append([image(is_vs_nrows_2),
                   text('This plot shows the insertion speed for %s columns.' % cols),
                   text('The method executescript seems to be the fastest when there are many columns.')])

    rows = 400 * 600
    maxcols = 100
    is_vs_ncols_1 = plot_insertion_speed_vs_ncols(rows, maxcols, 20, mvn1, mvn2)
    report.append([sheader('Insertion speed vs. number of columns'),
                   image(is_vs_ncols_1),
                   text('This plot shows the insertion speed for %s rows.' % rows),
                   text('For small number of columns the execute method is fastest, while for a larger'
                        ' number the executescript method is the fastest.')])

    # benchmark MAX_VARIABLE_NUMBER
    is_vs_mvn1 = plot_insertion_speed_vs_max_var_num()
    report.append([sheader('Insertion speed vs. MAX_VARIABLE_NUMBER'),
                   image(is_vs_mvn1),
                   text('This plot shows how the insertion speed of '),
                   codeline('cur.execute(query, values)'),
                   text('as implemented in sqlite_base. The execution time varies with MAX_VARIABLE_NUMBER.')])

    is_vs_mvn2 = benchmark_add_results_vs_MAX_VARIABLE_NUMBER()
    report.append([image(is_vs_mvn2),
                   text('This plot shows a real benchmark of existing code. The plot is specified by the following function.'),
                   codefunction(benchmark_add_results_vs_MAX_VARIABLE_NUMBER),
                   text('Wee see that a choice of 1000 instead of 250000 results in 50% faster insertion times')])



    with open('report.rst', 'w') as fil:
        for i in report:
            for j in i:
                fil.write(j)
    plt.show()

if __name__ == '__main__':
    make_report()

    #plot_insertion_speed_vs_max_var_num()

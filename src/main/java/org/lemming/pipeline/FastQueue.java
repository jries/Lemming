package org.lemming.pipeline;

import java.util.AbstractQueue;
import java.util.Collection;
import java.util.Iterator;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

import javolution.util.FastTable;

/**
 * a queue that uses the javalution FastTable as structure
 * @see javolution.util.FastTable
 * 
 * @author Ronny Sczech
 *
 * @param <E> - element type
 */
class FastQueue<E> extends AbstractQueue<E> implements BlockingQueue<E> {

	    /** The queued items */
	    private final FastTable<E> items;

	    /** Number of elements in the queue */
		private int count;

	    /*
	     * Concurrency control uses the classic two-condition algorithm
	     * found in any textbook.
	     */

	    /** Main lock guarding all access */
	    private final ReentrantLock lock;

	    /** Condition for waiting takes */
	    private final Condition notEmpty;

	    /** Condition for waiting puts */
	    private final Condition notFull;

		private int capacity;


	    /**
	     * Throws NullPointerException if argument is null.
	     *
	     * @param v the element
	     */
	    private static void checkNotNull(Object v) {
	        if (v == null)
	            throw new NullPointerException();
	    }

	    /**
	     * Inserts element at current put position, advances, and signals.
	     * Call only when holding lock.
	     */
	    private void enqueue(E x) {
	        final FastTable<E> items_ = this.items;
	        items_.addLast(x);
	        count++;
	        notEmpty.signal();
	    }

	    /**
	     * Extracts element at current take position, advances, and signals.
	     * Call only when holding lock.
	     */
	    private E dequeue() {
	        final FastTable<E> items_ = this.items;
	        E x = items_.pollFirst();
	        count--;
	        notFull.signal();
	        return x;
	    }

	    /**
	     * Creates an {@code ArrayBlockingQueue} with the given (fixed)
	     * capacity and default access policy.
	     *
	     * @param capacity the capacity of this queue
	     * @throws IllegalArgumentException if {@code capacity < 1}
	     */
	    public FastQueue(int capacity) {
	        this(capacity, true);
	    }

	    /**
	     * Creates an {@code ArrayBlockingQueue} with the given (fixed)
	     * capacity and the specified access policy.
	     *
	     * @param capacity the capacity of this queue
	     * @param fair if {@code true} then queue accesses for threads blocked
	     *        on insertion or removal, are processed in FIFO order;
	     *        if {@code false} the access order is unspecified.
	     * @throws IllegalArgumentException if {@code capacity < 1}
	     */
		private FastQueue(int capacity, boolean fair) {
	        if (capacity <= 0)
	            throw new IllegalArgumentException();
	        this.capacity = capacity;
	        this.items = new FastTable<>();
	        lock = new ReentrantLock(fair);
	        notEmpty = lock.newCondition();
	        notFull =  lock.newCondition();
	    }

	    /**
	     * Creates an {@code ArrayBlockingQueue} with the given (fixed)
	     * capacity, the specified access policy and initially containing the
	     * elements of the given collection,
	     * added in traversal order of the collection's iterator.
	     *
	     * @param capacity the capacity of this queue
	     * @param fair if {@code true} then queue accesses for threads blocked
	     *        on insertion or removal, are processed in FIFO order;
	     *        if {@code false} the access order is unspecified.
	     * @param c the collection of elements to initially contain
	     * @throws IllegalArgumentException if {@code capacity} is less than
	     *         {@code c.size()}, or less than 1.
	     * @throws NullPointerException if the specified collection or any
	     *         of its elements are null
	     */
	    public FastQueue(int capacity, boolean fair, Collection<? extends E> c) {
	        this(capacity, fair);

	        final ReentrantLock lock_ = this.lock;
	        lock_.lock(); // Lock only for visibility, not mutual exclusion
	        try {
	            int i = 0;
	            try {
	                for (E e : c) {
	                    checkNotNull(e);
	                    items.addLast(e);
	                    i++;
	                }
	            } catch (ArrayIndexOutOfBoundsException ex) {
	                throw new IllegalArgumentException();
	            }
	            count = i;
	        } finally {
	            lock_.unlock();
	        }
	    }

	    /**
	     * Inserts the specified element at the tail of this queue if it is
	     * possible to do so immediately without exceeding the queue's capacity,
	     * returning {@code true} upon success and throwing an
	     * {@code IllegalStateException} if this queue is full.
	     *
	     * @param e the element to add
	     * @return {@code true} (as specified by {@link Collection#add})
	     * @throws IllegalStateException if this queue is full
	     * @throws NullPointerException if the specified element is null
	     */
	    public boolean add(E e) {
	        return super.add(e);
	    }

	    /**
	     * Inserts the specified element at the tail of this queue if it is
	     * possible to do so immediately without exceeding the queue's capacity,
	     * returning {@code true} upon success and {@code false} if this queue
	     * is full.  This method is generally preferable to method {@link #add},
	     * which can fail to insert an element only by throwing an exception.
	     *
	     * @throws NullPointerException if the specified element is null
	     */
	    public boolean offer(E e) {
	        checkNotNull(e);
	        final ReentrantLock lock_ = this.lock;
	        lock_.lock();
	        try {
	            if (count == capacity)
	                return false;
				enqueue(e);
				return true;
	        } finally {
	            lock_.unlock();
	        }
	    }

	    /**
	     * Inserts the specified element at the tail of this queue, waiting
	     * for space to become available if the queue is full.
	     *
	     * @throws InterruptedException {@inheritDoc}
	     * @throws NullPointerException {@inheritDoc}
	     */
	    public void put(E e) throws InterruptedException {
	        checkNotNull(e);
	        final ReentrantLock lock_ = this.lock;
	        lock_.lockInterruptibly();
	        try {
	            while (count == capacity)
	                notFull.await();
	            enqueue(e);
	        } finally {
	        	lock_.unlock();
	        }
	    }

	    /**
	     * Inserts the specified element at the tail of this queue, waiting
	     * up to the specified wait time for space to become available if
	     * the queue is full.
	     *
	     * @throws InterruptedException {@inheritDoc}
	     * @throws NullPointerException {@inheritDoc}
	     */
	    public boolean offer(E e, long timeout, TimeUnit unit)
	        throws InterruptedException {

	        checkNotNull(e);
	        long nanos = unit.toNanos(timeout);
	        final ReentrantLock lock_ = this.lock;
	        lock_.lockInterruptibly();
	        try {
	            while (count == capacity) {
	                if (nanos <= 0)
	                    return false;
	                nanos = notFull.awaitNanos(nanos);
	            }
	            enqueue(e);
	            return true;
	        } finally {
	        	lock_.unlock();
	        }
	    }

	    public E poll() {
	        final ReentrantLock lock_ = this.lock;
	        lock_.lock();
	        try {
	            return (count == 0) ? null : dequeue();
	        } finally {
	        	lock_.unlock();
	        }
	    }

	    public E take() throws InterruptedException {
	        final ReentrantLock lock_ = this.lock;
	        lock_.lockInterruptibly();
	        try {
	            while (count == 0)
	                notEmpty.await();
	            return dequeue();
	        } finally {
	        	lock_.unlock();
	        }
	    }

	    public E poll(long timeout, TimeUnit unit) throws InterruptedException {
	        long nanos = unit.toNanos(timeout);
	        final ReentrantLock lock_ = this.lock;
	        lock_.lockInterruptibly();
	        try {
	            while (count == 0) {
	                if (nanos <= 0)
	                    return null;
	                nanos = notEmpty.awaitNanos(nanos);
	            }
	            return dequeue();
	        } finally {
	        	lock_.unlock();
	        }
	    }

	    public E peek() {
	        final ReentrantLock lock_ = this.lock;
	        lock_.lock();
	        try {
	            return items.peek(); // null when queue is empty
	        } finally {
	        	lock_.unlock();
	        }
	    }

	    // this doc comment is overridden to remove the reference to collections
	    // greater in size than Integer.MAX_VALUE
	    /**
	     * Returns the number of elements in this queue.
	     *
	     * @return the number of elements in this queue
	     */
	    public int size() {
	        final ReentrantLock lock_ = this.lock;
	        lock_.lock();
	        try {
	            return count;
	        } finally {
	        	lock_.unlock();
	        }
	    }

	    // this doc comment is a modified copy of the inherited doc comment,
	    // without the reference to unlimited queues.
	    /**
	     * Returns the number of additional elements that this queue can ideally
	     * (in the absence of memory or resource constraints) accept without
	     * blocking. This is always equal to the initial capacity of this queue
	     * less the current {@code size} of this queue.
	     *
	     * <p>Note that you <em>cannot</em> always tell if an attempt to insert
	     * an element will succeed by inspecting {@code remainingCapacity}
	     * because it may be the case that another thread is about to
	     * insert or remove an element.
	     */
	    public int remainingCapacity() {
	        final ReentrantLock lock_ = this.lock;
	        lock_.lock();
	        try {
	            return capacity - count;
	        } finally {
	        	lock_.unlock();
	        }
	    }

	    /**
	     * Removes a single instance of the specified element from this queue,
	     * if it is present.  More formally, removes an element {@code e} such
	     * that {@code o.equals(e)}, if this queue contains one or more such
	     * elements.
	     * Returns {@code true} if this queue contained the specified element
	     * (or equivalently, if this queue changed as a result of the call).
	     *
	     * <p>Removal of interior elements in circular array based queues
	     * is an intrinsically slow and disruptive operation, so should
	     * be undertaken only in exceptional circumstances, ideally
	     * only when the queue is known not to be accessible by other
	     * threads.
	     *
	     * @param o element to be removed from this queue, if present
	     * @return {@code true} if this queue changed as a result of the call
	     */
	    public boolean remove(Object o) {
	        if (o == null) return false;
	        final FastTable<E> items_ = this.items;
	        final ReentrantLock lock_ = this.lock;
	        lock_.lock();
	        try {
				return count > 0 && items_.remove(o);
			} finally {
	        	lock_.unlock();
	        }
	    }

	    /**
	     * Returns {@code true} if this queue contains the specified element.
	     * More formally, returns {@code true} if and only if this queue contains
	     * at least one element {@code e} such that {@code o.equals(e)}.
	     *
	     * @param o object to be checked for containment in this queue
	     * @return {@code true} if this queue contains the specified element
	     */
	    public boolean contains(Object o) {
	        if (o == null) return false;
	        final FastTable<E> items_ = this.items;
	        final ReentrantLock lock_ = this.lock;
	        lock_.lock();
	        try {
				return count > 0 && items_.contains(o);
			} finally {
	        	lock_.unlock();
	        }
	    }

	    /**
	     * Returns an array containing all of the elements in this queue, in
	     * proper sequence.
	     *
	     * <p>The returned array will be "safe" in that no references to it are
	     * maintained by this queue.  (In other words, this method must allocate
	     * a new array).  The caller is thus free to modify the returned array.
	     *
	     * <p>This method acts as bridge between array-based and collection-based
	     * APIs.
	     *
	     * @return an array containing all of the elements in this queue
	     */
	    public Object[] toArray() {
	        Object[] a;
	        final ReentrantLock lock_ = this.lock;
	        final FastTable<E> items_ = this.items;
	        lock_.lock();
	        try {
	            a = items_.toArray();
	        } finally {
	        	lock_.unlock();
	        }
	        return a;
	    }

	    /**
	     * Returns an array containing all of the elements in this queue, in
	     * proper sequence; the runtime type of the returned array is that of
	     * the specified array.  If the queue fits in the specified array, it
	     * is returned therein.  Otherwise, a new array is allocated with the
	     * runtime type of the specified array and the size of this queue.
	     *
	     * <p>If this queue fits in the specified array with room to spare
	     * (i.e., the array has more elements than this queue), the element in
	     * the array immediately following the end of the queue is set to
	     * {@code null}.
	     *
	     * <p>Like the {@link #toArray()} method, this method acts as bridge between
	     * array-based and collection-based APIs.  Further, this method allows
	     * precise control over the runtime type of the output array, and may,
	     * under certain circumstances, be used to save allocation costs.
	     *
	     * <p>Suppose {@code x} is a queue known to contain only strings.
	     * The following code can be used to dump the queue into a newly
	     * allocated array of {@code String}:
	     *
	     *  <pre> {@code String[] y = x.toArray(new String[0]);}</pre>
	     *
	     * Note that {@code toArray(new Object[0])} is identical in function to
	     * {@code toArray()}.
	     *
	     * @param a the array into which the elements of the queue are to
	     *          be stored, if it is big enough; otherwise, a new array of the
	     *          same runtime type is allocated for this purpose
	     * @return an array containing all of the elements in this queue
	     * @throws ArrayStoreException if the runtime type of the specified array
	     *         is not a supertype of the runtime type of every element in
	     *         this queue
	     * @throws NullPointerException if the specified array is null
	     */
	    public <T> T[] toArray(T[] a) {
	        final FastTable<E> items_ = this.items;
	        final ReentrantLock lock_ = this.lock;
	        lock_.lock();
	        try {
				items_.toArray(a);
			} finally {
	            lock_.unlock();
	        }
	        return a;
	    }

	    public String toString() {
	        final ReentrantLock lock_ = this.lock;
	        final FastTable<E> items_ = this.items;
	        lock_.lock();
	        try {
	            return items_.toString();     
	        } finally {
	            lock_.unlock();
	        }
	    }

	    /**
	     * Atomically removes all of the elements from this queue.
	     * The queue will be empty after this call returns.
	     */
	    public void clear() {
	        final FastTable<E> items_ = this.items;
	        final ReentrantLock lock_ = this.lock;
	        lock_.lock();
	        try {
	        	items_.clear();
	            int k = count;
	            if (k > 0) {
	                for (; k > 0 && lock_.hasWaiters(notFull); k--)
	                    notFull.signal();
	            }
	        } finally {
	            lock_.unlock();
	        }
	    }

	    /**
	     * @throws UnsupportedOperationException {@inheritDoc}
	     * @throws ClassCastException            {@inheritDoc}
	     * @throws NullPointerException          {@inheritDoc}
	     * @throws IllegalArgumentException      {@inheritDoc}
	     */
	    public int drainTo(Collection<? super E> c) {
	        return drainTo(c, Integer.MAX_VALUE);
	    }

	    /**
	     * @throws UnsupportedOperationException {@inheritDoc}
	     * @throws ClassCastException            {@inheritDoc}
	     * @throws NullPointerException          {@inheritDoc}
	     * @throws IllegalArgumentException      {@inheritDoc}
	     */
	    public int drainTo(Collection<? super E> c, int maxElements) {
	        checkNotNull(c);
	        if (c == this)
	            throw new IllegalArgumentException();
	        if (maxElements <= 0)
	            return 0;
	        final FastTable<E> items_ = this.items;
	        final ReentrantLock lock_ = this.lock;
	        lock_.lock();
	        try {
	            int n = Math.min(maxElements, count);
	            int i = 0;
	            try {
	                while (i < n) {
	                    E x = items_.poll();
	                    c.add(x);
	                    i++;
	                }
	                return n;
	            } finally {
	                // Restore invariants even if c.add() threw
	                if (i > 0) {
	                    count -= i;
	                    for (; i > 0 && lock_.hasWaiters(notFull); i--)
	                        notFull.signal();
	                }
	            }
	        } finally {
	            lock_.unlock();
	        }
	    }

		@Override
		public Iterator<E> iterator() {
			return items.iterator();
		}

		public Collection<E> view() {
			return items.unmodifiable();
		}

	    
	}

package org.lemming.pipeline;

import java.util.Collection;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import javolution.util.FastTable;

public class FastStore implements Store {
	
	private FastTable<Element> q = new FastTable<>();
	private final ReadWriteLock lock = new ReentrantReadWriteLock();
	
	/**
	 * 
	 */
	public FastStore(){
	}
	
	/**
	 * @return Returns the length of the queue.
	 */
	@Override
	public int getLength() {
		return q.size();
	}

	@Override
	public void put(Element el) {
		lock.writeLock().lock();
		try{
			q.addLast(el);
		} finally {
			lock.writeLock().unlock();
		}
	}
	
	@Override
	public Element peek(){
		Element res = null;
		lock.readLock().lock();
		try{
			while (!isEmpty()){
				res = q.peek();
				if (res != null) break;
			} 
		} finally {
			lock.readLock().unlock();
		}
		
		return res;
	}

	@Override
	public Element get() {
		Element res = null;
		lock.readLock().lock();
		try{
			while (!isEmpty()){
				res = q.poll();
				if (res != null) break;
			} 	
		} finally {
			lock.readLock().unlock();
		}
		
		return res;
	}

	@Override
	public boolean isEmpty() {
		return q.isEmpty();
	}
	
	@Override
	public  Collection<Element> view(){
		return q.immutable().value();
	}

}

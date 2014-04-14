package org.lemming.interfaces;

/**
 * A Well receives inputs and gives no outputs.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 * @param <T>
 */
public interface Well<T> {
	
        public void process(T object);
	public void beforeRun() {};
	public void afterRun() {};
}

package org.lemming.interfaces;

import org.lemming.queue.Store;

/**
 * A source creates new T into setOutput when run.  
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 * @param <T>
 */
public interface Source<T> {
        public void beforeRun();
        public void afterRun();
	
	/**
	 * Returns true if the Source has more outputs to be generated.
	 * @return
	 */
	public boolean hasMoreOutputs();

	/**
	 * Returns the generated object.
         *
         * May only be called when hasMoreOutputs() returned true.
	 * @return
	 */
	public T newOutput();
}

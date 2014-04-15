package org.lemming.queue;

import org.lemming.interfaces.Source;
import org.lemming.queue.NullStoreWarning;
import org.lemming.queue.Store;

/**
 * This class represents a module with a single output. This typically represents a generator of objects of type T.
 * It provides a standard implementation of the run method, which checks the output before calling newOutput
 * in a loop while hasMoreOutputs is true.
 * 
 * @author Thomas Pengo
 *
 * @param <T> Type parameter for the kind of objects that are being generated.
 */
public class SO<T> implements Runnable {

	Store<T> output;
        Source<T> source;

        public SO(Source<T> source, Store<T> output) {
            if (output==null)
                    throw new NullStoreWarning(this.getClass().getName()); 

            this.source = source;
            this.output = output;
        }

	@Override
	public final void run() {
                source.beforeRun();
		while (source.hasMoreOutputs()) {
			output.put(source.newOutput());
		}
                source.afterRun();
	}
	
}

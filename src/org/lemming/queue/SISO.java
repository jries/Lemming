package org.lemming.queue;

import org.lemming.interfaces.Processor;

public class SISO<T1,T2> implements Runnable {

	Store<T1> input;
	Store<T2> output;
        Processor<T1,T2> processor;

        public SISO(Store<T1> input, Processor<T1,T2> processor, Store<T2> output) {
            this.input = input;
            this.processor = processor;
            this.output = output;
        }

	@Override
	public void run() {
		if (input==null || output==null) {
			throw new NullStoreWarning(this.getClass().getName()); 
                }
		
		T1 loc;
		while ((loc=input.get())!=null) {
                        T2 result = processor.process(loc);
                        if (result != null) {
                            output.put(result);
                        }
		}
	}
	
}

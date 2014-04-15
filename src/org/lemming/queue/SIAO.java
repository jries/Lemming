package org.lemming.queue;

import java.util.AbstractList;

import org.lemming.interfaces.Processor;

public class SIAO<T1,T2> implements Runnable {

	Store<T1> input;
	Store<T2> output;
        Processor<T1,AbstractList<T2>> processor;

        public SIAO(Store<T1> input, Processor<T1,AbstractList<T2>> processor, Store<T2> output) {
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
                        AbstractList<T2> result = processor.process(loc);
                        if (result != null) {
                                for (T2 element : result) {
                                        output.put(element);
                                }
                        }
		}
	}
	
}

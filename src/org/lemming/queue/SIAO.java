package org.lemming.queue;

import org.lemming.interfaces.Processor;
import org.lemming.outputs.NullStoreWarning;
import org.lemming.queue.Store;

public class SIAO<T1,T2> implements Runnable {

	Store<T1> input;
	Store<T2> output;
        Processor<T1,Array<T2>> processor;

        SIAO(Processor<T1,Array<T2>> processor) {
            this->processor = processor;
        }

	@Override
	public void run() {
		if (input==null || output==null) {
			throw new NullStoreWarning(this.getClass().getName()); 
                }
		
		T1 loc;
		while ((loc=input.get())!=null) {
                        Array<T2> result = processor.process(loc);
                        if (result != nullptr) {
                                for (T2 element : result) {
                                        output.put(result);
                                }
                        }
		}
	}
	
	@Override
	public void setInput(Store<T1> s) {
		input = s;
	}

	@Override
	public void setOutput(Store<T2> s) {
		output = s;
	}

}

package org.lemming.pull_push;

import java.util.AbstractList;

import org.lemming.interfaces.Processor;
import org.lemming.interfaces.Well;

/**
 * A pushable processor is a well that processes objects and then
 * pushes the results to multiple downstream wells.
 * 
 * @param <T1> the input type
 * @param <T2> the output type
 */
public class PushableProcessor<T1, T2> implements Well<T1> {
        Processor<T1, T2> processor;
        AbstractList<Well<T2>> downstream;

        PushableProcessor(Processor<T1, T2> processor, AbstractList<Well<T2>> downstream) {
                this.processor = processor;
                this.downstream = downstream;
        }

        @Override
        public void beforeRun() {
                for (Well<T2> well : downstream) {
                        well.beforeRun();
                }
        }

        @Override
        public void afterRun() {
                for (Well<T2> well : downstream) {
                        well.afterRun();
                }
        }

        @Override
	public void process(T1 input) {
                T2 result = processor.process(input);
                if (result != null) {
                    for (Well<T2> well : downstream) {
                        well.process(result);
                    }
                }
        }
}

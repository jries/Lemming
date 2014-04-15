package org.lemming.pull_push;

import org.lemming.interfaces.Processor;
import org.lemming.interfaces.Source;

/**
 * A pullable processor is a source that pulls objects from a
 * upstream Source and processes them with the given processor.
 * 
 * @param <T1> the input type
 * @param <T2> the output type
 */
public class PullableProcessor<T1, T2> implements Source<T2> {
        Source<T1> upstream;
        Processor<T1, T2> processor;
        T2 next_output;

        PullableProcessor(Source<T1> upstream, Processor<T1, T2> processor) {
            this.upstream = upstream;
            this.processor = processor;
        }

        @Override
        public void beforeRun() {
            upstream.beforeRun();
        }

        public void afterRun() {
            upstream.afterRun();
        }

        @Override
	public boolean hasMoreOutputs() {
            while (next_output == null) {
                if (!upstream.hasMoreOutputs()) {
                    return false;
                }

                T1 input = upstream.newOutput();
                next_output = processor.process(input);
            }

            return true;
        }

        @Override
	public T2 newOutput() {
            T2 result = next_output;
            next_output = null;
            return result;
        }
}

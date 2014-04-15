package org.lemming.pull_push;

import java.util.AbstractList;
import java.util.Iterator;

import org.lemming.interfaces.Processor;
import org.lemming.interfaces.Source;

/**
 * A pullable array processor is a source that pulls objects from a
 * upstream Source and processes them with the given list-producing
 * processor.
 *
 * It unpacks the result list and provides the result as outputs.
 * 
 * @param <T1> the input type
 * @param <T2> the output type
 */
public class PullableListProcessor<T1, T2> implements Source<T2> {
        Source<T1> upstream;
        Processor<T1, AbstractList<T2>> processor;
        Iterator<T2> next_output = null;

        PullableListProcessor(Source<T1> upstream, Processor<T1, AbstractList<T2>> processor) {
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
            while (next_output == null || !next_output.hasNext()) {
                if (!upstream.hasMoreOutputs()) {
                    return false;
                }

                T1 input = upstream.newOutput();
                next_output = processor.process(input).iterator();
            }

            return true;
        }

        @Override
	public T2 newOutput() {
            return next_output.next();
        }
}

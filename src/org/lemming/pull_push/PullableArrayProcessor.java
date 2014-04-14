package org.lemming.pull_push;

/**
 * A pullable array processor is a source that pulls objects from a
 * upstream Source and processes them with the given array-producing
 * processor.
 *
 * It unpacks the result array and provides the result as outputs.
 * 
 * @param <T1> the input type
 * @param <T2> the output type
 */
public PullableProcessor<T1, T2> implements Source<T2> {
        Source<T1> upstream;
        Processor<T1, Array<T2>> processor;
        Array<T2> array = null;
        int array_index = 0;

        PullableProcessor(Source<T1> upstream, Processor<T1, Array<T2>> processor) {
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
            return upstream.hasMoreOutputs();
        }

        @Override
	public T2 newOutput() {
            while (array == null) {
                T1 input = upstream.newOutput();
                if (input == null) {
                    return null;
                } else {
                    array = processor.process(input);
                    array_index = 0;
                }
            }

            T2 result = array[array_index];
            array_index += 1;
            if (array_index == array.length()) {
                array = null;
            }
            return result;
        }
}

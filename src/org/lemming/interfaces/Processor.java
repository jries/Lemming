package org.lemming.interfaces;

/**
 * A processor is a module that transforms T1s into T2s.
 *
 * The processor may return null to indicate that the operation
 * had no results.
 * 
 * @param <T1> the input type
 * @param <T2> the output type
 */
public interface Processor<T1, T2> {
	public abstract T2 process(T1 input);
}

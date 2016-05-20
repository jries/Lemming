package org.lemming.interfaces;

import java.util.Queue;

import net.imglib2.type.numeric.RealType;

/**
 * interface for detector plug-ins which uses some pre-processing
 * 
 * @author Ronny Sczech
 *
 * @param <T> - data type
 */
public interface PreProcessor<T extends RealType<T>> extends Detector<T>{
	
	Frame<T> preProcess(final Queue<Frame<T>> list, final boolean isLast);
	
	int getNumberOfFrames();
}

package org.lemming.pipeline;

import java.util.concurrent.LinkedBlockingQueue;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Store;

/**
 * a queue using a linked list as structure
 * 
 * @author Ronny Sczech
 *
 */
public class LinkedStore extends LinkedBlockingQueue<Element> implements Store {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5362955780141552726L;

	public LinkedStore(int capacity) {
		super(capacity);
	}

}
